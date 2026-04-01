import math
import torch

import logging
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.text.symbols import N_VOCAB
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import duration_loss, sequence_mask
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
        self.mas_const = -0.5 * LOG_2_PI * n_feats

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

        # Use MAS to find most likely alignment `attn` between text and fine mel-spectrogram
        with torch.no_grad():
            # This computes the distance between every text token and ground truth mel frame 
            # using a  Gaussian log-likelihood formula 
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_fine_square = torch.matmul(factor.transpose(1, 2), y_fine ** 2)
            y_fine_mu_double = torch.matmul(-mu_x.transpose(1, 2), y_fine)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_fine_square - y_fine_mu_double + mu_square + self.mas_const

            attn_fine = maximum_path_gpu(log_prior, attn_mask_fine.squeeze(1).to(torch.int32), log_prior.dtype)

        mas_durations = torch.sum(attn_fine.unsqueeze(1), -1).squeeze(1)  # (B, T_text)
        
        # I am adding a 2 to make the values greater than 1, because MSE is more forgiving with sub-unitary losses
        # and more punishing with supra-unitary losses.
        # E.g. 0.6 ** 2 < 0.6, but 1.6 ** 2 > 1.6  
        # This helps Duration Predictor A LOT. We have to compensate for the +2 before synthesis, see inference.py.  
        logw_ = torch.log(2 + mas_durations.unsqueeze(1)) * x_mask

        # logw - log-scaled durations from the Duration Predictor
        # logw_ - log-scaled durations calculated by the Monotonic Alignment Search algorithm. 
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Expand encoder mel vectors into a fine resolution predicted mel using the fine alignment
        mu_y_fine = torch.matmul(mu_x, attn_fine.squeeze(1))

        if self.prior_loss:
            prior_loss = torch.sum(torch.abs(y_fine - mu_y_fine) * y_fine_mask)
            prior_loss = prior_loss / (torch.sum(y_fine_mask) * self.n_feats)
        else:
            prior_loss = 0

        # Downsample fine resolution predicted mel to standard resolution for the decoder
        mu_y_coarse = torch.nn.functional.avg_pool1d(mu_y_fine, kernel_size=2, stride=2)
        detached_mu_y_coarse = mu_y_coarse.detach()

        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)

        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=detached_mu_y_coarse)

        return diff_loss, dur_loss, prior_loss
