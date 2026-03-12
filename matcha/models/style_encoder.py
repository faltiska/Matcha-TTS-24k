import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.model import sequence_mask
from super_monotonic_align import maximum_path as maximum_path_gpu

log = logging.getLogger(__name__)

LOG_2_PI = math.log(2 * math.pi)


def masked_mean_pool(x, mask):
    """Mean pool x (B, C, T) over time using mask (B, 1, T)."""
    x = x * mask
    return x.sum(dim=2) / mask.sum(dim=2).clamp(min=1)


class AcousticStyleEncoder(nn.Module):
    """Predicts encoder_speaker_embedding from mel spectrogram.

    Takes mel (B, n_feats, T_mel) and produces a single vector (B, spk_emb_dim_enc).
    Architecture: stack of Conv1d+ReLU layers, masked mean pool, linear projection.
    """

    def __init__(self, n_feats, hidden_channels, n_layers, spk_emb_dim_enc):
        super().__init__()
        self.convs = nn.ModuleList()
        in_ch = n_feats
        for _ in range(n_layers):
            self.convs.append(nn.Conv1d(in_ch, hidden_channels, kernel_size=5, padding=2))
            in_ch = hidden_channels
        self.proj = nn.Linear(hidden_channels, spk_emb_dim_enc)

    def forward(self, mel, mel_mask):
        """
        Args:
            mel:      (B, n_feats, T_mel)
            mel_mask: (B, 1, T_mel)
        Returns:
            (B, spk_emb_dim_enc)
        """
        x = mel
        for conv in self.convs:
            x = torch.relu(conv(x * mel_mask))
        pooled = masked_mean_pool(x, mel_mask)
        return self.proj(pooled)


class RhythmStyleEncoder(nn.Module):
    """Predicts duration_speaker_embedding from ASE embedding.

    Takes:
        ase_emb: (B, spk_emb_dim_enc) — from AcousticStyleEncoder

    Produces: (B, spk_emb_dim_dur)
    """

    def __init__(self, spk_emb_dim_enc, spk_emb_dim_dur, **kwargs):
        super().__init__()
        self.proj = nn.Linear(spk_emb_dim_enc, spk_emb_dim_dur)

    def forward(self, ase_emb):
        """
        Args:
            ase_emb: (B, spk_emb_dim_enc)
        Returns:
            (B, spk_emb_dim_dur)
        """
        return self.proj(ase_emb)


class StyleEncoderLightningModule(LightningModule):
    """Trains AcousticStyleEncoder and RhythmStyleEncoder against a frozen MatchaTTS checkpoint.

    For each batch:
      1. Run frozen Matcha encoder with real embeddings → mu_x_real, logw_ (MAS), x_dp, x_mask
      2. ASE predicts encoder_speaker_embedding from mel
      3. RSE predicts duration_speaker_embedding from ASE output
      4. Run frozen Matcha encoder twice (once per loss, with the other embedding detached)
         → mu_x_pred (for ASE loss), logw_pred (for RSE loss)
      5. Losses: MSE(mu_x_pred, mu_x_real) + MSE(logw_pred, logw_mas)
    """

    def __init__(
        self,
        matcha_checkpoint_path,
        n_feats,
        n_channels,
        ase_hidden_channels,
        ase_n_layers,
        spk_emb_dim_enc,
        spk_emb_dim_dur,
        optimizer=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.acoustic_style_encoder = AcousticStyleEncoder(
            n_feats=n_feats,
            hidden_channels=ase_hidden_channels,
            n_layers=ase_n_layers,
            spk_emb_dim_enc=spk_emb_dim_enc,
        )
        self.rhythm_style_encoder = RhythmStyleEncoder(
            spk_emb_dim_enc=spk_emb_dim_enc,
            spk_emb_dim_dur=spk_emb_dim_dur,
        )

        matcha = MatchaTTS.load_from_checkpoint(matcha_checkpoint_path, map_location="cpu", weights_only=False)
        matcha.eval()
        for param in matcha.parameters():
            param.requires_grad = False
        self.matcha = matcha

        self.mas_const = -0.5 * LOG_2_PI * n_feats

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())

    def _run_matcha_encoder_with_mas(self, x, x_lengths, y, y_lengths, encoder_spk_emb, duration_spk_emb):
        """Run Matcha text encoder + MAS with given embeddings.

        Returns mu_x, logw (predicted durations), logw_ (MAS ground truth), x_dp, x_mask, y_mask.
        Replicates TextEncoder.forward internals to capture x_dp before it is discarded.
        """
        encoder = self.matcha.encoder

        x_emb = encoder.emb(x) * math.sqrt(encoder.n_channels)
        x_emb = torch.transpose(x_emb, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_emb.size(2)), 1).to(x_emb.dtype)
        x_emb = encoder.prenet(x_emb, x_mask)
        if self.matcha.n_spks > 1:
            x_emb = torch.cat([x_emb, encoder_spk_emb.unsqueeze(-1).repeat(1, 1, x_emb.shape[-1])], dim=1)
        x_out = encoder.encoder(x_emb, x_mask)
        mu_x = encoder.proj_m(x_out) * x_mask
        x_dp = (x_out[:, :-encoder.spk_emb_dim, :] if self.matcha.n_spks > 1 else x_out).detach()
        logw = encoder.proj_w(x_dp, x_mask, duration_spk_emb)

        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
        y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
        y_mu_double = torch.matmul(-mu_x.transpose(1, 2), y)
        mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
        log_prior = y_square - y_mu_double + mu_square + self.mas_const
        attn = maximum_path_gpu(log_prior, attn_mask.squeeze(1).to(torch.int32), log_prior.dtype)

        mas_durations = torch.sum(attn.unsqueeze(1), -1).squeeze(1)
        logw_ = torch.log(1e-8 + mas_durations.unsqueeze(1)) * x_mask

        return mu_x, logw, logw_, x_dp, x_mask, y_mask

    def _compute_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]

        real_enc_emb = self.matcha.encoder_speaker_embeddings(spks)
        real_dur_emb = self.matcha.duration_speaker_embeddings(spks)

        # Run frozen Matcha encoder + MAS with real embeddings
        with torch.no_grad():
            mu_x_real, _, logw_mas, x_dp, x_mask, y_mask = self._run_matcha_encoder_with_mas(
                x, x_lengths, y, y_lengths, real_enc_emb, real_dur_emb
            )

        # ASE predicts encoder embedding from mel
        pred_enc_emb = self.acoustic_style_encoder(y, y_mask)

        # RSE predicts duration embedding from ASE output.
        # Detach pred_enc_emb so RSE loss does not flow back into ASE weights.
        pred_dur_emb = self.rhythm_style_encoder(pred_enc_emb.detach())

        # Run encoder twice so each loss only trains its own encoder.
        # ASE loss: pred_dur_emb detached so ase_loss cannot reach RSE weights.
        mu_x_pred, _, _ = self.matcha.encoder(x, x_lengths, pred_enc_emb, pred_dur_emb.detach())
        # RSE loss: pred_enc_emb detached so rse_loss cannot reach ASE weights.
        _, logw_pred, _ = self.matcha.encoder(x, x_lengths, pred_enc_emb.detach(), pred_dur_emb)

        ase_loss = F.mse_loss(mu_x_pred, mu_x_real)
        rse_loss = F.mse_loss(logw_pred, logw_mas)

        with torch.no_grad():
            ase_emb_dist = torch.mean((pred_enc_emb - real_enc_emb) ** 2).sqrt()
            rse_emb_dist = torch.mean((pred_dur_emb - real_dur_emb) ** 2).sqrt()

        return ase_loss, rse_loss, ase_emb_dist, rse_emb_dist

    def training_step(self, batch, batch_idx):
        ase_loss, rse_loss, ase_emb_dist, rse_emb_dist = self._compute_losses(batch)
        total_loss = ase_loss + rse_loss
        bs = batch["x"].shape[0]
        self.log_dict({
            "loss/train": total_loss,
            "loss/train_ase": ase_loss,
            "loss/train_rse": rse_loss,
            "emb_dist/train_ase": ase_emb_dist,
            "emb_dist/train_rse": rse_emb_dist,
            "step": float(self.global_step),
        }, on_step=False, on_epoch=True, logger=True, batch_size=bs)
        return total_loss

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
