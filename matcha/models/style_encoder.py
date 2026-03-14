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
    """Trains AcousticStyleEncoder against a frozen MatchaTTS checkpoint.

    For each batch:
      1. Run frozen Matcha encoder with real speaker embedding → mu_x_real
      2. ASE predicts speaker embedding from mel
      3. Run frozen Matcha encoder with predicted embedding → mu_x_pred
      4. Loss: L1(mu_x_pred, mu_x_real)
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

        self.acoustic_style_encoder = StyleEncoder(
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

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())

    def _compute_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]

        real_spk_emb = self.matcha.encoder_speaker_embeddings(spks)

        y_mask = sequence_mask(y_lengths, y.shape[-1]).unsqueeze(1).to(y.dtype)

        with torch.no_grad():
            mu_x_real, _, _ = self.matcha.encoder(x, x_lengths, real_spk_emb, real_spk_emb)

        pred_spk_emb = self.acoustic_style_encoder(y, y_mask)

        mu_x_pred, _, _ = self.matcha.encoder(x, x_lengths, pred_spk_emb, pred_spk_emb)

        loss = F.l1_loss(mu_x_pred, mu_x_real)

        with torch.no_grad():
            emb_dist = torch.mean((pred_spk_emb - real_spk_emb) ** 2).sqrt()

        return loss, emb_dist

    def training_step(self, batch, batch_idx):
        ase_loss, emb_dist = self._compute_losses(batch)
        bs = batch["x"].shape[0]
        self.log_dict({
            "loss/train": ase_loss,
            "emb_dist/train": emb_dist,
            "step": float(self.global_step),
        }, on_step=False, on_epoch=True, logger=True, batch_size=bs)
        return ase_loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log("grad_norm/grad_2.0_norm_total", norms["grad_2.0_norm_total"], on_step=True, on_epoch=False, logger=True)
        per_param_norms = torch.stack([p.detach().norm() for p in self.parameters()])
        total_param_norm = torch.linalg.vector_norm(per_param_norms)
        self.log("grad_norm/param_norm", total_param_norm, on_step=True, on_epoch=False, logger=True)

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, 'create_batches'):
            old_len = len(sampler)
            sampler.create_batches()
            new_len = len(sampler)
            if old_len != new_len:
                log.error(f"Batch count changed from {old_len} to {new_len} at epoch {self.current_epoch}, this will cause Lightning to stop running validation.")
