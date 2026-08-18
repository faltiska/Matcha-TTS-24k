import math
import torch
import torch.nn.functional as F
import logging
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.text.symbols import N_VOCAB
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import box_downsample, sequence_mask, LOG_DURATION_OFFSET
from matcha.utils.perceptual_mel_weights import build_perceptual_mel_weights
from super_monotonic_align import maximum_path as maximum_path_gpu 

log = logging.getLogger(__name__)

LOG_2_PI = math.log(2 * math.pi)


class MatchaTTS(BaseLightningClass):  # 🍵
    def __init__(
        self,
        n_spks,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
        spk_emb_dim=None,
        optimizer=None, # parameter required by BaseLightningClass
        scheduler=None, # parameter required by BaseLightningClass
        prior_loss=True,
        duration_loss=True,
        log_diagnostics=True,
        prior_loss_threshold=0.08,
        duration_loss_threshold=0.15,
        plot_mel_on_validation_end=False,
        sample_rate=None,
        f_min=None,
        f_max=None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = N_VOCAB
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.prior_loss = prior_loss
        self.duration_loss = duration_loss
        self.plot_mel_on_validation_end = plot_mel_on_validation_end

        # Fixed, per-mel-bin weight that biases the prior loss towards the frequency range where human
        # hearing is most sensitive (see build_perceptual_mel_weights). Shaped (1, n_feats, 1) so it
        # broadcasts directly over (batch, n_feats, time) mels. The leading dimension of size 1 stands in
        # for a future per-speaker dimension (n_spks, n_feats, 1), selected by speaker id, once per-speaker
        # weights are precomputed from each speaker's own data.
        perceptual_mel_weights = build_perceptual_mel_weights(n_feats, sample_rate, f_min, f_max)
        self.register_buffer("perceptual_mel_weights", perceptual_mel_weights.view(1, n_feats, 1), persistent=False)

        if n_spks > 1:
            self.speaker_embeddings_enc = torch.nn.Embedding(n_spks, spk_emb_dim)
            self.speaker_embeddings_dur = torch.nn.Embedding(n_spks, spk_emb_dim)

        self.encoder = TextEncoder(
            encoder.encoder_params,
            encoder.duration_predictor_params,
            N_VOCAB,
            spk_emb_dim,
        )

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
        )

        self.encoder = torch.compile(self.encoder, dynamic=True)
        self.decoder.estimator = torch.compile(self.decoder.estimator, dynamic=True)

        self.update_data_statistics(data_statistics)

    def forward(self, x, x_lengths, y, y_lengths, y_fine, y_fine_lengths, spks, is_training_step):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms at hop=256.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms at hop=256.
                shape: (batch_size,)
            y_fine (torch.Tensor): batch of corresponding mel-spectrograms at hop=128.
                shape: (batch_size, n_feats, max_mel_length * 2)
            y_fine_lengths (torch.Tensor): lengths of mel-spectrograms at hop=128.
                shape: (batch_size,)
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        speaker_embedding_enc = self.speaker_embeddings_enc(spks)
        speaker_embedding_dur = self.speaker_embeddings_dur(spks)

        # Skip building the backward graph for the encoder when prior_loss and duration_loss are set to false; it speeds
        # up the forward pass.
        with torch.set_grad_enabled(self.prior_loss or self.duration_loss):
            # Get encoder_outputs `mu_x` and log-scaled token durations `logw`.
            mu_x, logw, x_mask = self.encoder(x, x_lengths, speaker_embedding_enc, speaker_embedding_dur)

        y_fine_max_length = y_fine.shape[-1]
        y_fine_mask = sequence_mask(y_fine_lengths, y_fine_max_length).unsqueeze(1).to(x_mask)
        attn_mask_fine = x_mask.unsqueeze(-1) * y_fine_mask.unsqueeze(2)

        attn_fine = self.find_alignment(attn_mask_fine, mu_x, y_fine)

        if self.duration_loss:
            # torch.sum(attn.unsqueeze(1), -1)) says how many mel frames each text token aligns to
            # x_mask has 1s for valid text tokens, 0s for padding positions, to ensure loss is only calculated on 
            # valid tokens, preventing attention to padding.
            mas_durations = torch.sum(attn_fine.unsqueeze(1), -1).squeeze(1)  # (B, T_text)
            
            # x_mask has 1s for valid text tokens, 0s for padding positions, to ensure loss is only calculated on
            # valid tokens, preventing attention to padding.

            # At small x values, the log curve is very steep.
            # If MAS made an error finding 2 frames instead of 1, the log function jumps by a lot.
            # If duration from mas is 7 instead of 8, the log jumps much less.
            # Considering the phonemization scheme and the fact that we use 5.3ms frames, many durations found by MAS are
            # small numbers, and the log function reacts too much to them. By adding an offset, we move into the more linear
            # part of the log curve. Inference subtracts the same value, so the real durations are unchanged.
            # This helps the Duration Predictor learn, by a lot.
            logw_ = torch.log(LOG_DURATION_OFFSET + mas_durations.unsqueeze(1)) * x_mask

            # logw - log-scaled durations from the Duration Predictor
            # logw_ - log-scaled durations calculated by the Monotonic Alignment Search algorithm.
            threshold = self.hparams.duration_loss_threshold
            dur_loss = F.huber_loss(logw, logw_, delta=threshold, reduction='sum') / torch.sum(x_mask)
            # Original code was pure MSE:
            # dur_loss = torch.sum((logw - logw_) ** 2) / torch.sum(x_mask)
            # but it leads to a huge gap between the validation and the train losses.

            if self.hparams.log_diagnostics:
                with torch.no_grad():
                    self._log_diagnostics(logw, logw_, x_mask, self.METRIC_DURATION, is_training_step)
        else:
            dur_loss = 0

        # Original code was: 
        #   mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        #   mu_y = mu_y.transpose(1, 2)
        # but that can be simplified as:
        #   mu_y = torch.matmul(mu_x, attn.squeeze(1))
        mu_y_fine = torch.matmul(mu_x, attn_fine.squeeze(1))

        if self.prior_loss:
            # Original code was: 
            #   prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            # but I could remove the constants without affecting the meaning of the loss.
            #   prior_loss = torch.sum(((y - mu_y) ** 2) * y_mask)
            threshold = self.hparams.prior_loss_threshold
            # perceptual_mel_weights biases the loss towards the mel bins the human ear is most sensitive
            # to, without adding a second competing loss term (see build_perceptual_mel_weights).
            y_fine_weighted = y_fine * y_fine_mask * self.perceptual_mel_weights
            mu_y_fine_weighted = mu_y_fine * y_fine_mask * self.perceptual_mel_weights
            prior_loss = F.huber_loss(y_fine_weighted, mu_y_fine_weighted, delta=threshold, reduction='sum')
            prior_loss = prior_loss / torch.sum(y_fine_mask)

            if self.hparams.log_diagnostics:
                with torch.no_grad():
                    self._log_diagnostics(y_fine, mu_y_fine, y_fine_mask.expand_as(y_fine), self.METRIC_PRIOR, is_training_step)
        else:
            prior_loss = 0

        # noinspection PyCallingNonCallable
        mu_y = box_downsample(mu_y_fine)
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)

        # Detach mu_y to prevent Decoder gradients from flowing back to the Encoder. We do not want 
        # the Encoder to learn to produce mels that make the Decoder's job easier. 
        # We want the Encoder to learn how to produce mels that match the ground truth.
        diff_loss = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y.detach())

        return diff_loss, dur_loss, prior_loss

    def find_alignment(self, attn_mask_fine, mu_x, y_fine):
        # Use MAS to find most likely alignment `attn` between text and fine mel-spectrogram
        # It computes the distance between every text token and ground truth mel frame. 
        with torch.no_grad():
            # Original code was using a factor defined as a tensor:
            #   factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            # but it is equivalent to multiplying by -0.5 directly.
            # It was also adding a mas constant: 
            #   self.mas_const = -0.5 * LOG_2_PI * n_feats
            # but that does not influence the alignment at all. 
            # It was also using matmuls and transpositions that can be simplified as follows:
            y_sq = -0.5 * (y_fine ** 2).sum(dim=1, keepdim=True)  # (B, 1, T_mel)
            mu_y = torch.matmul(mu_x.transpose(1, 2), y_fine)  # (B, T_text, T_mel)
            mu_sq = -0.5 * (mu_x ** 2).sum(dim=1, keepdim=True).transpose(1, 2)  # (B, T_text, 1)
            log_prior = y_sq + mu_y + mu_sq  # (B, T_text, T_mel)
            attn_fine = maximum_path_gpu(log_prior, attn_mask_fine.squeeze(1).to(torch.int32), log_prior.dtype)

        return attn_fine
