"""
I have modified Matcha TTS and just completed a training that finally sounds good.

The model has an Encoder with Speaker Embeddings, that accepts a phonetic representation, and generates a state 
that will be used in 2 ways:
- as input for the Duration Predictor
- as input for the Mel Predictor

The output of the Mel Predictor combined with durations from a MAS algorithm results in a predicted mel spectrogram 
which will be used to calculate the Encoder loss and as input for the diffusion based Decoder.
The Encoder loss (called a prior loss) is calculated as the difference between the ground truth mel and the predicted mel.
Duration Predictor and Decoder losses are decoupled from the Encoder. Only the prior loss drives the Encoder and the 
Speaker Embeddings. The architecture works fine, and the synthesized voice sounds good.

On top of this, I wrote a Style Encoder, which was supposed to be able to predict Speaker Embeddings for new speakers, 
based on a small set of mel spectrograms.

I trained that, and ran inference with it on a new speaker. The speaker sounds intelligible, timbre is very accurate, 
but it has a weird accent and makes pronunciation mistakes.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.model import sequence_mask

log = logging.getLogger(__name__)


def masked_mean_pool(x, mask):
    """Mean pool x (B, C, T) over time using mask (B, 1, T)."""
    x = x * mask
    return x.sum(dim=2) / mask.sum(dim=2).clamp(min=1)


class StyleEncoder(nn.Module):
    """Predicts speaker embedding from mel spectrogram.

    Takes mel (B, n_feats, T_mel) and produces a single vector (B, spk_emb_dim).
    Architecture: stack of Conv1d+ReLU layers, masked mean pool, linear projection.
    """

    def __init__(self, n_feats, hidden_channels, n_layers, spk_emb_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        in_ch = n_feats
        for _ in range(n_layers):
            self.convs.append(nn.Conv1d(in_ch, hidden_channels, kernel_size=5, padding=2))
            in_ch = hidden_channels
        self.proj = nn.Linear(hidden_channels, spk_emb_dim)

    def forward(self, mel, mel_mask):
        """
        Args:
            mel:      (B, n_feats, T_mel)
            mel_mask: (B, 1, T_mel)
        Returns:
            (B, spk_emb_dim)
        """
        x = mel
        for conv in self.convs:
            x = torch.relu(conv(x * mel_mask))
        pooled = masked_mean_pool(x, mel_mask)
        return self.proj(pooled)


class StyleEncoderLightningModule(LightningModule):
    """Trains the StyleEncoder against a frozen MatchaTTS checkpoint.

    For each batch:
      1. The StyleEncoder predicts speaker embedding from mel
      2. Run the frozen Matcha encoder with real speaker embedding → mu_x_real, logw_real
      3. Run the frozen Matcha encoder with predicted embedding → mu_x_pred, logw_pred
      4. Calculate losses: 
            Acoustic loss: smooth L1(mu_x_pred, mu_x_real)
            Rhythm loss: smooth L1(logw_pred, logw_real)
    """

    def __init__(
        self,
        matcha_checkpoint_path,
        n_feats,
        ase_hidden_channels,
        ase_n_layers,
        spk_emb_dim,
        optimizer=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.style_encoder = StyleEncoder(
            n_feats=n_feats,
            hidden_channels=ase_hidden_channels,
            n_layers=ase_n_layers,
            spk_emb_dim=spk_emb_dim,
        )

        matcha = MatchaTTS.load_from_checkpoint(matcha_checkpoint_path, map_location="cpu", weights_only=False)
        matcha.eval()
        for param in matcha.parameters():
            param.requires_grad = False
        self.matcha = matcha
        self.matcha.encoder = torch.compile(self.matcha.encoder)
        self.style_encoder = torch.compile(self.style_encoder)
        self.register_buffer("_quantile_probs", torch.tensor([0.25, 0.5, 0.75, 0.9]), persistent=False)

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())

    def _compute_losses(self, batch, batch_idx):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y_fine, y_fine_lengths = batch["y_fine"], batch["y_fine_lengths"]
        spks = batch["spks"]

        y_fine_mask = sequence_mask(y_fine_lengths, y_fine.shape[-1]).unsqueeze(1).to(y_fine.dtype)
        pred_speaker_emb = self.style_encoder(y_fine, y_fine_mask)

        real_speaker_emb = self.matcha.speaker_embeddings(spks)
        with torch.no_grad():
            mu_x_real, logw_real, x_mask = self.matcha.encoder(x, x_lengths, real_speaker_emb)

        mu_x_pred, logw_pred, _ = self.matcha.encoder(x, x_lengths, pred_speaker_emb)

        # Acoustic loss - gradients flow back to pred_speaker_emb
        acoustic_loss = F.smooth_l1_loss(mu_x_pred * x_mask, mu_x_real * x_mask, beta=0.001, reduction='sum')
        acoustic_loss = acoustic_loss / x_mask.sum()

        # Rhythm loss - gradients flow back to pred_speaker_emb
        rhythm_loss = F.smooth_l1_loss(logw_pred * x_mask, logw_real * x_mask, beta=0.001, reduction='sum')
        rhythm_loss = rhythm_loss / x_mask.sum()

        # Combined loss
        total_loss = acoustic_loss + rhythm_loss

        with torch.no_grad():
            per_sample_emb_dist = (pred_speaker_emb - real_speaker_emb).pow(2).mean(dim=1).sqrt()
            emb_dist = per_sample_emb_dist.mean()

            is_first_batch_of_epoch = batch_idx == 0
            if self.training and is_first_batch_of_epoch:
                valid_residuals = torch.abs(mu_x_pred - mu_x_real)[x_mask.expand_as(mu_x_pred).bool()]
                q = torch.quantile(valid_residuals, self._quantile_probs)
                self.log("debug/residuals_p25", q[0], on_step=False, on_epoch=True, batch_size=x.shape[0])
                self.log("debug/residuals_p50", q[1], on_step=False, on_epoch=True, batch_size=x.shape[0])
                self.log("debug/residuals_p75", q[2], on_step=False, on_epoch=True, batch_size=x.shape[0])
                self.log("debug/residuals_p90", q[3], on_step=False, on_epoch=True, batch_size=x.shape[0])

                q_emb = torch.quantile(per_sample_emb_dist, self._quantile_probs)
                self.log("debug/emb_dist_p25", q_emb[0], on_step=False, on_epoch=True, batch_size=x.shape[0])
                self.log("debug/emb_dist_p50", q_emb[1], on_step=False, on_epoch=True, batch_size=x.shape[0])
                self.log("debug/emb_dist_p75", q_emb[2], on_step=False, on_epoch=True, batch_size=x.shape[0])
                self.log("debug/emb_dist_p90", q_emb[3], on_step=False, on_epoch=True, batch_size=x.shape[0])

        return total_loss, acoustic_loss, rhythm_loss, emb_dist

    def training_step(self, batch, batch_idx):
        total_loss, acoustic_loss, rhythm_loss, emb_dist = self._compute_losses(batch, batch_idx)
        bs = batch["y_fine"].shape[0]
        self.log_dict({
            "train/total_loss": total_loss,
            "train/acoustic_loss": acoustic_loss,
            "train/rhythm_loss": rhythm_loss,
            "train/emb_dist": emb_dist,
        }, on_step=False, on_epoch=True, logger=True, batch_size=bs)
        return total_loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log("norm/grad_2.0_norm_total", norms["grad_2.0_norm_total"], on_step=True, on_epoch=False, logger=True)
        per_param_norms = torch.stack([p.detach().norm() for p in self.parameters()])
        total_param_norm = torch.linalg.vector_norm(per_param_norms)
        self.log("norm/param_norm", total_param_norm, on_step=True, on_epoch=False, logger=True)

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, 'create_batches'):
            old_len = len(sampler)
            sampler.create_batches()
            new_len = len(sampler)
            if old_len != new_len:
                log.error(f"Batch count changed from {old_len} to {new_len} at epoch {self.current_epoch}, this will cause Lightning to stop running validation.")