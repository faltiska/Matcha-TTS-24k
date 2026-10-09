import copy
import math
import torch
import torch.nn.functional as F
import logging
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.text.symbols import N_VOCAB
from matcha.models.components.flow_matching import CFM
from matcha.models.components.decoder_ema import paired_flow_matching_losses
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import DEFAULT_DOWNSAMPLER, generate_path, get_downsampler, sequence_mask, LOG_DURATION_OFFSET
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
        downsampler=DEFAULT_DOWNSAMPLER,
        sample_rate=None,
        f_min=None,
        f_max=None,
        decoder_ema_decay=None,
        alignment_jitter=True,
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
        perceptual_mel_weights = build_perceptual_mel_weights(n_feats, f_min, f_max)
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

        # A frozen copy of the Decoder holding the exponential moving average of its weights (see decoder_ema.py).
        # It is copied before compilation, so its state_dict keys carry no _orig_mod. It is never called: validation
        # swaps its weights into the live Decoder. Its parameters never require gradients, so the optimizer groups
        # and the optimizer state of older checkpoints stay the same.
        # It is registered after the trained modules, so it does not shift their position in named_parameters().
        self.decoder_ema = None
        if decoder_ema_decay is not None:
            self.decoder_ema = copy.deepcopy(self.decoder.estimator)
            self.decoder_ema.requires_grad_(False)

        self.encoder = torch.compile(self.encoder, dynamic=True)
        self.decoder.estimator = torch.compile(self.decoder.estimator, dynamic=True)

        # The mel downsample sits between the two compiled regions above, so it used to run eagerly as
        # a handful of separate pointwise kernels. Compiling it on its own fuses the padding, the
        # strided slices and the scaling into a single kernel, measured at roughly 2-3x faster than
        # eager on training shapes. It stays a region of its own because forward() cannot become one
        # single region: find_alignment calls the third-party MAS kernel and _log_diagnostics calls
        # Lightning's logger and indexes with a boolean mask, and each of those forces a graph break.
        # Which filter runs here is the `downsampler` hyperparameter; see get_downsampler.
        self.downsample = torch.compile(get_downsampler(downsampler), dynamic=True)

        self.update_data_statistics(data_statistics)

    def forward(self, x, x_lengths, y, y_lengths, y_fine, y_fine_lengths, spks, is_training_step):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.
        and, on validation steps only, a fourth reading of the flow matching loss pinned to the end
        of the trajectory (last 5% of it). Not calculated if is_training_step = true.
        When the Decoder keeps an exponential moving average of its weights, validation steps also return
        both flow matching readings for the averaged weights, drawn from the same timesteps and noise.

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

        # Set alignment_jitter to false to condition the Decoder on the exact MAS alignment.
        # The jitter sits after the duration and prior losses, so the Duration Predictor still learns the true MAS
        # durations and the Encoder still learns each phoneme's own frame, not a blend with its neighbours.
        # mu_x is detached because the Decoder's input is detached below anyway, so building a backward
        # graph through this second assembly would only cost memory.
        should_jitter_alignment = is_training_step and self.hparams.alignment_jitter
        if should_jitter_alignment:
            attn_fine_jittered = self.jitter_alignment(attn_fine, attn_mask_fine.squeeze(1))
            mu_y_fine = torch.matmul(mu_x.detach(), attn_fine_jittered)

        mu_y = self.downsample(mu_y_fine)
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)

        # Detach mu_y to prevent Decoder gradients from flowing back to the Encoder. We do not want 
        # the Encoder to learn to produce mels that make the Decoder's job easier. 
        # We want the Encoder to learn how to produce mels that match the ground truth.
        decoder_condition = mu_y.detach()
        if is_training_step:
            diff_loss = self.decoder.compute_loss(x1=y, mask=y_mask, mu=decoder_condition)
            return diff_loss, dur_loss, prior_loss, None, None

        if self.decoder_ema is None:
            diff_loss = self.decoder.compute_loss(x1=y, mask=y_mask, mu=decoder_condition)
            # A second reading of the same loss, pinned to the end of the trajectory, for validation only.
            late_diff_loss = self.decoder.compute_loss(
                    x1=y,
                    mask=y_mask,
                    mu=decoder_condition,
                sample_late_trajectory=True,
                )
            return diff_loss, dur_loss, prior_loss, late_diff_loss, None

        (diff_loss, late_diff_loss), ema_diff_losses = paired_flow_matching_losses(
            self.decoder, list(self.decoder_ema.parameters()), y, y_mask, decoder_condition
        )
        return diff_loss, dur_loss, prior_loss, late_diff_loss, ema_diff_losses

    # The fraction of its own frames a token may hand over to a neighbour, never less than one frame.
    # One fine frame is 5.3ms, half a frame at the resolution the Decoder works at.
    # The allowance follows duration because the Duration Predictor's error does: measured against MAS on
    # this corpus it is 0.4 to 0.9 frames for tokens up to 8 frames, which is 88% of them, and 1.4 to 2.6
    # frames on the pauses above 13, where the worst tenth reaches 5. A flat allowance would under-jitter
    # exactly the tokens the predictor gets most wrong.
    JITTER_FRACTION = 0.15

    def jitter_alignment(self, attn_fine, attn_mask_fine):
        """
        Moves each token boundary, so the mel the Decoder is conditioned on comes out slightly different on
        every step.

        With the prior loss off, nothing trains the Encoder, so MAS returns the same alignment every epoch
        and the Decoder keeps seeing one fixed conditioning mel per sample, which it can memorise. A moving
        boundary also matches what inference feeds the Decoder, where durations come from the Duration
        Predictor and are never exactly the MAS ones.

        The ground truth mel does not move, so the frame total may not change either. Frames are therefore
        only traded between two neighbouring tokens: what one loses, the other gains. Tokens are paired up
        and only the boundary inside a pair moves. The pairing starts one token later on odd steps, so
        every boundary gets its turn.
        """
        durations = attn_fine.sum(-1)
        first_token = self.global_step % 2
        pair_count = (durations.shape[1] - first_token) // 2
        last_token = first_token + 2 * pair_count

        first_durations = durations[:, first_token:last_token:2]
        second_durations = durations[:, first_token + 1:last_token:2]

        frames_the_first_can_give = self.frames_a_token_can_give(first_durations)
        frames_the_second_can_give = self.frames_a_token_can_give(second_durations)
        # A uniform draw over the integers in [-frames_the_first_can_give, frames_the_second_can_give].
        # A positive draw moves frames from the second token to the first. torch.randint cannot do this,
        # because it takes one pair of bounds for the whole tensor and here every pair has its own.
        uniform_draw = torch.rand(first_durations.shape, device=durations.device, dtype=durations.dtype)
        draw_span = frames_the_first_can_give + frames_the_second_can_give + 1
        transferred_frames = torch.floor(uniform_draw * draw_span) - frames_the_first_can_give

        # A pair that touches padding trades nothing. The padding has no frames to give, and a frame handed
        # to a padded token is masked out of the path, which would leave a frame of silence in the mel.
        both_tokens_are_real = (first_durations >= 1) & (second_durations >= 1)
        transferred_frames = transferred_frames * both_tokens_are_real

        jittered_durations = durations.clone()
        jittered_durations[:, first_token:last_token:2] += transferred_frames
        jittered_durations[:, first_token + 1:last_token:2] -= transferred_frames

        return generate_path(jittered_durations, attn_mask_fine)

    def frames_a_token_can_give(self, durations):
        """
        How many frames each token may hand to a neighbour: its share of JITTER_FRACTION, but at least one
        frame, since a frame is the smallest move there is, and never its last frame, because MAS gives
        every real token at least one and inference also floors predicted durations at one.
        Padding holds no frames and reports nothing to give.
        """
        allowance = (durations * self.JITTER_FRACTION).round().clamp(min=1)
        return torch.minimum(durations - 1, allowance).clamp(min=0)

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
