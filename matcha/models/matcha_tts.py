import math
import torch
import torch.nn.functional as F
import logging
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.text.symbols import N_VOCAB
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import duration_loss, sequence_mask, downsample
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
        plot_mel_on_validation_end=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = N_VOCAB
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.prior_loss = prior_loss
        self.plot_mel_on_validation_end = plot_mel_on_validation_end

        if n_spks > 1:
            self.speaker_embeddings = torch.nn.Embedding(n_spks, spk_emb_dim)

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

        self.update_data_statistics(data_statistics)
        self.batch_idx = 0
        self.register_buffer("_quantile_probs", torch.tensor([0.25, 0.5, 0.75, 0.9]), persistent=False)

    def forward(self, x, x_lengths, y, y_lengths, y_fine, y_fine_lengths, spks):
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
        speaker_embedding = self.speaker_embeddings(spks)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, speaker_embedding)
        y_fine_max_length = y_fine.shape[-1]

        y_fine_mask = sequence_mask(y_fine_lengths, y_fine_max_length).unsqueeze(1).to(x_mask)
        attn_mask_fine = x_mask.unsqueeze(-1) * y_fine_mask.unsqueeze(2)

        with torch.autocast(device_type="cuda", enabled=False):
            # I want these 2 in fp32 because they are involved in matmul operations down below
            # I think bf16 doesn't have enough precision to distinguish between two competing alignment paths whose 
            # scores are very close, and MAS suddenly finds a different path at some point, after some 100 epochs.
            # Prior loss shoots up by 1% which may be fine, but duration loss shoots up by 60%. 
            # I always saw prior loss shooting up at som point, but the duration loos effect is new, probably related 
            # to introducing the super-resolution mechanism. 
            mu_x = mu_x.float()
            y_fine = y_fine.float() 
            attn_fine = self.find_alignment(attn_mask_fine, mu_x, y_fine)

        # torch.sum(attn.unsqueeze(1), -1)) says how many mel frames each text token aligns to
        # x_mask has 1s for valid text tokens, 0s for padding positions, to ensure loss is only calculated on 
        # valid tokens, preventing attention to padding.
        mas_durations = torch.sum(attn_fine.unsqueeze(1), -1).squeeze(1)  # (B, T_text)
        
        # I am adding a 2 to make the log-space values greater than 1, because MSE is more forgiving with sub-unitary
        # losses, and more punishing with supra-unitary losses.
        # E.g. 0.6 ** 2 < 0.6, but 1.6 ** 2 > 1.6  
        # This helps Duration Predictor A LOT. We have to compensate for the +2 before synthesis, see inference.py.  
        logw_ = torch.log(2 + mas_durations.unsqueeze(1)) * x_mask

        # Original code was: 
        #   mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        #   mu_y = mu_y.transpose(1, 2)
        # but that can be simplified as:
        #   mu_y = torch.matmul(mu_x, attn.squeeze(1))
        mu_y_fine = torch.matmul(mu_x, attn_fine.squeeze(1))

        if self.prior_loss:
            # logw - log-scaled durations from the Duration Predictor
            # logw_ - log-scaled durations calculated by the Monotonic Alignment Search algorithm.
            # It only makes sense to calculate the duration loss if prior loss is set to true.
            # Otherwise, MAS would find the same alignment and Duration Predictor has nothing new to learn. 
            dur_loss = duration_loss(logw, logw_, x_lengths)
    
            # Original code was: 
            #   prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            # but I could remove the constants without affecting the meaning of the loss.
            #   prior_loss = torch.sum(((y - mu_y) ** 2) * y_mask)
            # The L2 loss was causing instability in early epochs, so I switched to a smooth L1 loss:
            prior_loss = F.smooth_l1_loss(y_fine * y_fine_mask, mu_y_fine * y_fine_mask, beta=0.04, reduction='sum')
            prior_loss = prior_loss / torch.sum(y_fine_mask)
            # It punishes errors larger than 0.04 like an L1, but is lenient like an L2 with smaller errors. 
            # Huber loss will be much smaller in value, as it multiplies by the threshold, so stick to smooth_l1.

            # This helps pick a good beta value: train for 50 epochs and look at thr p50 distribution. Say it has values
            # between 0.03 and 0.05. That tells you a beta of 0.04 is probably best. All errors below the threshold are
            # given an L2 treatment, all values above it an L1.
            # Also, if the distribution has a heavy right tail (p90 much larger than 3x p50), it's a sign some frames 
            # are consistently hard to predict.
            if self.batch_idx == 0 and self.training:
                with torch.no_grad():
                    valid_residuals = torch.abs(y_fine - mu_y_fine)[y_fine_mask.expand_as(y_fine).bool()]
                    q = torch.quantile(valid_residuals, self._quantile_probs)
                    self.log("debug_prior/residuals_p25", q[0], on_step=False, on_epoch=True, batch_size=y.shape[0])
                    self.log("debug_prior/residuals_p50", q[1], on_step=False, on_epoch=True, batch_size=y.shape[0])
                    self.log("debug_prior/residuals_p75", q[2], on_step=False, on_epoch=True, batch_size=y.shape[0])
                    self.log("debug_prior/residuals_p90", q[3], on_step=False, on_epoch=True, batch_size=y.shape[0])
        else:
            prior_loss = 0
            dur_loss = 0

        mu_y_coarse = downsample(mu_y_fine)

        # Detach mu_y to prevent Decoder gradients from flowing back to the Encoder. We do not want 
        # the Encoder to learn to produce mels that make the Decoder's job easier. 
        # We want the Encoder to learn how to produce mels that match the ground truth.
        detached_mu_y_coarse = mu_y_coarse.detach()
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        diff_loss = self.decoder.compute_loss(x1=y, mask=y_mask, mu=detached_mu_y_coarse)

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
