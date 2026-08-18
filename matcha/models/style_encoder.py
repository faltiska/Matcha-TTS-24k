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


def masked_stats_pool(x, mask):
    """Mean+std pool x (B, C, T) over time using mask (B, 1, T). Returns (B, 2C).

    Concatenates the masked mean and the masked standard deviation over time, so each clip is
    summarized by both its average voice characteristics and how much they vary within the clip.
    """
    n = mask.sum(dim=2).clamp(min=1)
    x_masked = x * mask
    mean = x_masked.sum(dim=2) / n
    mean_sq = (x_masked * x).sum(dim=2) / n
    std = (mean_sq - mean ** 2).clamp(min=1e-8).sqrt()
    return torch.cat([mean, std], dim=1)


def masked_smooth_l1_loss(prediction, target, mask, beta):
    """Return Smooth L1 loss averaged over valid output elements only."""
    element_losses = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    valid_elements = mask.expand_as(prediction)
    valid_element_count = valid_elements.sum().clamp(min=1)
    return (element_losses * valid_elements).sum() / valid_element_count


def effective_embedding_error(embedding_error, speaker_projection):
    """How much a speaker embedding error actually changes what the main model does.

    A speaker embedding reaches the main model through exactly one linear layer, which turns it into the
    scale and shift values applied inside the encoder or the duration predictor. That layer responds
    strongly to a handful of embedding directions and barely at all to the rest. In a v22 checkpoint,
    113 of the duration predictor's 128 input directions carry under 5% of the effect of the strongest
    one, and 101 of 128 do so in the text encoder. We should measure the effect again on a production grade ckpt.

    Measuring the plain distance between two embeddings therefore overstates any error sitting in a
    direction the model ignores, and a large plain distance can leave the model's output untouched. That
    reading is misleading enough to draw wrong conclusions from.

    Pushing the error through that same layer reports it in the units the model reacts to, and weights
    every direction by how much it actually matters. The layer's bias cancels out in a difference, so
    only its weight is needed.

    Args:
        embedding_error:    (B, spk_emb_dim) predicted embedding minus the real one
        speaker_projection: the frozen Linear layer that consumes the speaker embedding
    Returns:
        (B,) per-sample error, as the RMS change in the scale and shift values the main model applies.
    """
    modulation_error = F.linear(embedding_error, speaker_projection.weight)
    return modulation_error.pow(2).mean(dim=1).sqrt()


class SpeakerEmbeddingPredictor(nn.Module):
    """Predicts one speaker embedding from a mel spectrogram.

    Takes mel (B, n_feats, T_mel) and produces a single vector (B, spk_emb_dim).
    Architecture: stack of Conv1d+SiLU layers, masked mean+std pool, linear projection.
    """

    def __init__(self, n_feats, hidden_channels, n_layers, spk_emb_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        in_ch = n_feats
        for _ in range(n_layers):
            self.convs.append(nn.Conv1d(in_ch, hidden_channels, kernel_size=5, padding=2))
            in_ch = hidden_channels
        self.proj = nn.Linear(hidden_channels * 2, spk_emb_dim)

    def forward(self, mel, mel_mask):
        """
        Args:
            mel:      (B, n_feats, T_mel)
            mel_mask: (B, 1, T_mel)
        Returns:
            (B, spk_emb_dim) speaker embedding
        """
        x = mel
        for conv in self.convs:
            x = F.silu(conv(x * mel_mask))
        pooled = masked_stats_pool(x, mel_mask)
        return self.proj(pooled)


class StyleEncoder(nn.Module):
    """Predicts both of the main model's speaker embeddings from a mel spectrogram.

    Each embedding is predicted by its own independent network, with nothing shared between them. This
    mirrors the strict separation the main model enforces between its Encoder and its Duration Predictor,
    where neither loss can influence the other's weights.

    A single shared stack with two projection heads would be cheaper, but the acoustic and the rhythm loss
    would both train that stack, so each would shape the features the other depends on. It would also give
    the two embeddings less independent capacity, which is the reason StyleTTS2 uses two entirely separate
    encoder networks.
    """

    def __init__(self, n_feats, hidden_channels, n_layers, spk_emb_dim):
        super().__init__()
        self.acoustic_predictor = SpeakerEmbeddingPredictor(n_feats, hidden_channels, n_layers, spk_emb_dim)
        self.rhythm_predictor = SpeakerEmbeddingPredictor(n_feats, hidden_channels, n_layers, spk_emb_dim)

    def forward(self, mel, mel_mask):
        """
        Args:
            mel:      (B, n_feats, T_mel)
            mel_mask: (B, 1, T_mel)
        Returns:
            emb_enc: (B, spk_emb_dim) - embedding for the text encoder
            emb_dur: (B, spk_emb_dim) - embedding for the duration predictor
        """
        return self.acoustic_predictor(mel, mel_mask), self.rhythm_predictor(mel, mel_mask)


class StyleEncoderLightningModule(LightningModule):
    """Trains the StyleEncoder against a frozen MatchaTTS checkpoint.

    For each batch:
      1. The StyleEncoder predicts both speaker embeddings from mel
      2. Run the frozen Matcha encoder with both real embeddings → mu_x_real, logw_real
      3. Run it with the predicted encoder embedding → mu_x_pred
      4. Run it with the real encoder embedding and the predicted rhythm embedding → logw_pred.
         Using the real encoder embedding here keeps acoustic error out of the rhythm loss; see the
         comment in _compute_losses for why that matters.
      5. Calculate losses:
            Acoustic loss: smooth L1(mu_x_pred, mu_x_real), both perceptually weighted per mel bin
            Rhythm loss: smooth L1(logw_pred, logw_real)

    The two losses are added into one number for the optimizer but do not interact, the same way the main
    model's three losses do not. Each one trains only its own predictor network: the acoustic loss cannot
    reach the rhythm network and the rhythm loss cannot reach the acoustic one, because the two networks
    share no weights and step 4 keeps acoustic error out of the rhythm path.

    Each loss is averaged over its own valid output elements, so the 100 acoustic mel values per phoneme do
    not outvote the single rhythm value by sheer count.
    """

    # Absolute error percentiles reported to Tensorboard. The set matches the main model's diagnostics.
    # The trailing 1.0 is the largest error in the batch, which is what the Huber thresholds are tuned to.
    QUANTILE_PROBS = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    QUANTILE_LABELS = ["p25", "p50", "p75", "p90", "p95", "p99", "p100"]

    def __init__(
        self,
        matcha_checkpoint_path,
        ase_hidden_channels,
        ase_n_layers,
        acoustic_loss_threshold=0.002,
        rhythm_loss_threshold=0.004,
        optimizer=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        matcha = MatchaTTS.load_from_checkpoint(matcha_checkpoint_path, map_location="cpu", weights_only=False)
        matcha.eval()
        for param in matcha.parameters():
            param.requires_grad = False
        del matcha.decoder
        self.matcha = matcha

        self.style_encoder = StyleEncoder(
            n_feats=matcha.n_feats,
            hidden_channels=ase_hidden_channels,
            n_layers=ase_n_layers,
            spk_emb_dim=matcha.spk_emb_dim,
        )

        print("[🍵] Compiling the Style Encoder...")
        self.style_encoder = torch.compile(self.style_encoder, dynamic=True)
        self.register_buffer("_quantile_probs", torch.tensor(self.QUANTILE_PROBS), persistent=False)
        self._print_embedding_distance_reference()

    def _speaker_projections(self):
        """The two frozen layers a speaker embedding passes through to reach the main model.

        One feeds the text encoder, the other the duration predictor. self.matcha.encoder is
        torch.compile()'d, so its submodules are reached through _orig_mod.
        """
        text_encoder = self.matcha.encoder._orig_mod
        return text_encoder.encoder.spk_proj, text_encoder.proj_w.spk_proj

    def _print_embedding_distance_reference(self):
        """Print how far apart two different trained speakers are, in the units the distance charts use.

        Without a reference the acoustic_emb_dist and rhythm_emb_dist charts are hard to read. This value
        is the yardstick: an error approaching it means the prediction is as wrong as picking a different
        speaker, while an error far below it means the main model barely notices.
        """
        encoder_projection, duration_projection = self._speaker_projections()
        encoder_table = self.matcha.speaker_embeddings_enc.weight.detach()
        duration_table = self.matcha.speaker_embeddings_dur.weight.detach()

        for label, table, projection in (
            ("acoustic", encoder_table, encoder_projection),
            ("rhythm", duration_table, duration_projection),
        ):
            differences = table.unsqueeze(0) - table.unsqueeze(1)
            distances = effective_embedding_error(differences.flatten(0, 1), projection)
            speaker_pairs_are_distinct = distances > 0
            mean_distance = distances[speaker_pairs_are_distinct].mean()
            print(f"[🍵] Two different speakers differ by {mean_distance:.4f} on the {label}_emb_dist chart.")

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "matcha"):
            self.matcha.eval()
        return self

    def configure_optimizers(self):
        """Build the optimizer with biases excluded from weight decay.

        This mirrors BaseLightningClass.configure_optimizers, which the main model uses. AdamW's weight
        decay pulls every parameter it is applied to towards zero. For the two projection biases that is
        actively harmful: those biases are what lets a predicted embedding reach the non-zero centre of
        the trained speaker embedding tables, so decaying them biases every prediction towards the origin
        of the embedding space.

        The Style Encoder currently has no normalization layers, so a bias name check is enough to
        separate the two groups.
        """
        decayed_params, undecayed_params = [], []
        for parameter_name, parameter in self.style_encoder.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter_is_a_bias = parameter_name.endswith("bias")
            if parameter_is_a_bias:
                undecayed_params.append(parameter)
            else:
                decayed_params.append(parameter)

        return self.hparams.optimizer(params=[
            {"params": decayed_params},
            {"params": undecayed_params, "weight_decay": 0.0},
        ])

    def _compute_losses(self, batch, batch_idx):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y_fine, y_fine_lengths = batch["y_fine"], batch["y_fine_lengths"]
        spks = batch["spks"]

        y_fine_mask = sequence_mask(y_fine_lengths, y_fine.shape[-1]).unsqueeze(1).to(y_fine.dtype)
        pred_speaker_emb_enc, pred_speaker_emb_dur = self.style_encoder(y_fine, y_fine_mask)

        real_speaker_emb_enc = self.matcha.speaker_embeddings_enc(spks)
        real_speaker_emb_dur = self.matcha.speaker_embeddings_dur(spks)
        with torch.no_grad():
            mu_x_real, logw_real, x_mask = self.matcha.encoder(x, x_lengths, real_speaker_emb_enc, real_speaker_emb_dur)

        # Acoustic path. Only the encoder embedding shapes the mel output, so the duration output of this
        # pass is discarded; the rhythm embedding is passed along only because the signature requires it.
        mu_x_pred, _, _ = self.matcha.encoder(x, x_lengths, pred_speaker_emb_enc, pred_speaker_emb_dur)

        # Rhythm path, deliberately fed the REAL encoder embedding alongside the predicted rhythm embedding.
        #
        # The Duration Predictor no longer reads the raw phoneme embeddings; it reads the Encoder output,
        # detached. If we took the durations from the acoustic path above, the Duration Predictor's input
        # would already carry whatever error the predicted encoder embedding has, while the detach would
        # stop the rhythm gradient from ever reaching the encoder head that caused it. The rhythm head would
        # then be the only place that gradient can land, so it would learn a vector biased to cancel an
        # acoustic error rather than one that describes the speaker's rhythm. That compensation is unstable,
        # because it goes stale as soon as the acoustic head improves, and it does not survive add_speaker.py
        # averaging the two predictions independently across recordings.
        #
        # Pairing the predicted rhythm embedding with the real encoder embedding removes the contamination,
        # so this loss measures rhythm head error and nothing else. The cost is one extra pass through the
        # frozen encoder. Nothing in that pass requires grad except the predicted rhythm embedding, so
        # autograd only builds a graph through the Duration Predictor.
        _, logw_pred, _ = self.matcha.encoder(x, x_lengths, real_speaker_emb_enc, pred_speaker_emb_dur)

        # The main model's prior loss weights every mel bin by perceptual_mel_weights before measuring error,
        # biasing it towards the frequencies human hearing is most sensitive to. The frozen encoder we are
        # distilling was trained under that weighting, so the same weighting is applied here. Without it the
        # Style Encoder would spread its limited capacity evenly across all mel bins and trade error between
        # them differently than the model it has to reproduce.
        weighted_mu_x_pred = mu_x_pred * self.matcha.perceptual_mel_weights
        weighted_mu_x_real = mu_x_real * self.matcha.perceptual_mel_weights

        # Acoustic loss - gradients flow back to pred_speaker_emb_enc
        acoustic_loss = masked_smooth_l1_loss(
            weighted_mu_x_pred, weighted_mu_x_real, x_mask, beta=self.hparams.acoustic_loss_threshold
        )

        # Rhythm loss - gradients flow back to pred_speaker_emb_dur
        rhythm_loss = masked_smooth_l1_loss(
            logw_pred, logw_real, x_mask, beta=self.hparams.rhythm_loss_threshold
        )

        # The two losses are summed only so a single number can be handed to the optimizer. They do not
        # interact: each one trains its own predictor network and nothing else.
        total_loss = acoustic_loss + rhythm_loss

        with torch.no_grad():
            encoder_projection, duration_projection = self._speaker_projections()
            per_sample_emb_dist_enc = effective_embedding_error(
                pred_speaker_emb_enc - real_speaker_emb_enc, encoder_projection
            )
            per_sample_emb_dist_dur = effective_embedding_error(
                pred_speaker_emb_dur - real_speaker_emb_dur, duration_projection
            )
            emb_dist_enc = per_sample_emb_dist_enc.mean()
            emb_dist_dur = per_sample_emb_dist_dur.mean()

            is_first_batch_of_epoch = batch_idx == 0
            if self.training and is_first_batch_of_epoch:
                bs = x.shape[0]
                # error_quantiles/acoustic_* and error_quantiles/rhythm_* are the per-element absolute
                # errors that drive acoustic_loss and rhythm_loss: how far the frozen Matcha encoder's mel
                # output and predicted durations drift when fed a predicted speaker embedding instead of a
                # real one. The acoustic errors are measured on the perceptually weighted tensors, the same
                # ones the loss sees, so acoustic_p100 can be read directly as the acoustic_loss_threshold.
                acoustic_errors = torch.abs(weighted_mu_x_pred - weighted_mu_x_real)[x_mask.expand_as(mu_x_pred).bool()]
                self._log_quantiles("acoustic", acoustic_errors, bs)
                rhythm_errors = torch.abs(logw_pred - logw_real)[x_mask.bool()]
                self._log_quantiles("rhythm", rhythm_errors, bs)

                # error_quantiles/acoustic_emb_dist_* and error_quantiles/rhythm_emb_dist_* measure how far
                # the predicted speaker embeddings land from the trained lookup table entries, expressed as
                # the change they cause in the scale and shift values the main model applies rather than as
                # a plain distance (see effective_embedding_error for why the plain distance misleads).
                # They are not optimized. Nothing pulls a predicted embedding towards the stored one: the
                # Style Encoder is meant to learn only from the effect an embedding has on the main model,
                # which automatically weights each embedding direction by how much it matters.
                self._log_quantiles("acoustic_emb_dist", per_sample_emb_dist_enc, bs)
                self._log_quantiles("rhythm_emb_dist", per_sample_emb_dist_dur, bs)

        return total_loss, acoustic_loss, rhythm_loss, emb_dist_enc, emb_dist_dur

    def _log_quantiles(self, name, values, batch_size):
        """Log the absolute error distribution, from the 25th percentile up to the maximum.

        The quantile set matches BaseLightningClass._log_diagnostics, so the Style Encoder's thresholds can
        be tuned the same way the main model's are: the threshold is set to the largest absolute error seen
        at convergence, which is the p100 entry. A threshold placed there makes the Huber loss clip only
        while training is still unstable and errors are large, and leaves it purely quadratic once the model
        has converged.

        Training runs in bf16-mixed precision, and torch.quantile() rejects bfloat16 inputs, so the
        percentiles are read straight out of the sorted error tensor instead.
        """
        no_valid_values = values.numel() == 0
        if no_valid_values:
            return

        sorted_errors = torch.sort(values).values
        last_index = sorted_errors.numel() - 1
        quantile_indices = (self._quantile_probs * last_index).long().clamp(0, last_index)
        quantiles = sorted_errors[quantile_indices]
        for label, quantile in zip(self.QUANTILE_LABELS, quantiles):
            self.log(f"error_quantiles/{name}_{label}", quantile, on_step=False, on_epoch=True, batch_size=batch_size)

    def _log_losses(self, prefix, total_loss, acoustic_loss, rhythm_loss, emb_dist_enc, emb_dist_dur, batch_size):
        self.log_dict({
            f"{prefix}/style_loss": total_loss,
            f"{prefix}/acoustic_style_loss": acoustic_loss,
            f"{prefix}/rhythm_style_loss": rhythm_loss,
            f"{prefix}/acoustic_emb_dist": emb_dist_enc,
            f"{prefix}/rhythm_emb_dist": emb_dist_dur,
        }, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        total_loss, acoustic_loss, rhythm_loss, emb_dist_enc, emb_dist_dur = self._compute_losses(batch, batch_idx)
        bs = batch["y_fine"].shape[0]
        self._log_losses("train", total_loss, acoustic_loss, rhythm_loss, emb_dist_enc, emb_dist_dur, bs)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, acoustic_loss, rhythm_loss, emb_dist_enc, emb_dist_dur = self._compute_losses(batch, batch_idx)
        bs = batch["y_fine"].shape[0]
        self._log_losses("val", total_loss, acoustic_loss, rhythm_loss, emb_dist_enc, emb_dist_dur, bs)
        return total_loss

    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self, norm_type=2)
    #     self.log("norm/grad_2.0_norm_total", norms["grad_2.0_norm_total"], on_step=True, on_epoch=False, logger=True)
    #     per_param_norms = torch.stack([p.detach().norm() for p in self.parameters()])
    #     total_param_norm = torch.linalg.vector_norm(per_param_norms)
    #     self.log("norm/param_norm", total_param_norm, on_step=True, on_epoch=False, logger=True)

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, 'create_batches'):
            old_len = len(sampler)
            sampler.create_batches()
            new_len = len(sampler)
            if old_len != new_len:
                log.error(f"Batch count changed from {old_len} to {new_len} at epoch {self.current_epoch}, this will cause Lightning to stop running validation.")
