import math
import torch

from matcha import utils
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import (
    duration_loss,
    generate_path,
    sequence_mask,
)
from super_monotonic_align import maximum_path as maximum_path_gpu 

log = utils.get_pylogger(__name__)

LOG_2_PI = math.log(2 * math.pi)

class MatchaTTS(BaseLightningClass):  # 🍵
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
        optimizer=None, # parameter required by BaseLightningClass
        scheduler=None, # parameter required by BaseLightningClass
        prior_loss=True,
        use_precomputed_durations=False,
        plot_mel_on_validation_end=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations
        self.plot_mel_on_validation_end = plot_mel_on_validation_end
        self.mas_const = -0.5 * LOG_2_PI * n_feats

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        self.update_data_statistics(data_statistics)

    def forward(self, x, x_lengths, y, y_lengths, spks=None, durations=None):
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
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            with torch.no_grad():
                # This computes the distance between every text token and ground truth mel frame 
                # using a  Gaussian log-likelihood formula 
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
                # Original code was:
                #   y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                # But (2.0 * factor * mu_x) is useless, because factor = -0.5 * tensor.ones
                # I've replaced it with just -mu_x
                y_mu_double = torch.matmul(-mu_x.transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + self.mas_const

                # Alternative: this computes pairwise distance between every text token and ground truth mel frame
                # Using L1-based MAS scoring would match the L1 formula I use for prior loss
                # log_prior = -torch.cdist(mu_x.transpose(1, 2), y.transpose(1, 2), p=1)

                # the GPU impl is about 5% faster
                attn = maximum_path_gpu(log_prior, attn_mask.squeeze(1).to(torch.int32), log_prior.dtype)
                # attn = monotonic_align.maximum_path_cpu(log_prior, attn_mask.squeeze(1))

        # torch.sum(attn.unsqueeze(1), -1)) says how many mel frames each text token aligns to
        # x_mask has 1s for valid text tokens, 0s for padding positions, to ensure loss is only calculated on 
        # valid tokens, preventing attention to padding.
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        # logw - log-scaled durations calculated by the TextEncoder's duration predictor
        # logw_ - log-scaled durations calculated the Monotonic Alignment Search algorithm. 
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        # That can be simplified as mu_y = torch.matmul(mu_x, attn.squeeze(1))

        # Detach mu_y to prevent diffusion gradients from flowing back to the encoder. We do not want 
        # the Encoder to learn to produce mels that make the Decoder's job easier. We want the Encoder to learn 
        # how to produce mels that match the ground truth.
        # Diffusion still learns from the diff_loss, it is not affected by this detach.
        detached_mu_y = mu_y.detach()

        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=detached_mu_y, spks=spks)

        if self.prior_loss:
            # Original code was: 
            #   prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            # but I could remove the constants without affecting the meaning of the loss.
            #   prior_loss = torch.sum(((y - mu_y) ** 2) * y_mask)
            # Also, the values are too small, 0.05 the after first epoch tending to 0.0001
            # Such small gradients are not allowing the model to learn, so I am not squaring them up anymore.
            # This had a positive effect on duration estimation too, which started showing much smaller losses.  
            # Before this change, I could not get duration loss below 0.15, not even 400K steps.
            prior_loss = torch.sum(torch.abs(y - mu_y) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return diff_loss, dur_loss, prior_loss
