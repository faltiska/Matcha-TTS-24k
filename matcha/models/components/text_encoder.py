import math

import torch
import torch.nn as nn

from matcha.utils.model import sequence_mask
from matcha.models.components.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        # Assumes x is a 3D tensor (batch, channels, time).
        x = x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        return x


class ConvSiluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.silu_drop = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.silu_drop.append(nn.Sequential(nn.SiLU(), nn.Dropout(p_dropout)))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
            self.silu_drop.append(nn.Sequential(nn.SiLU(), nn.Dropout(p_dropout)))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.silu_drop[i](x)
        x = x_org + self.proj(x)
        return x * x_mask

class DurationPredictor(nn.Module):
    """Predicts phoneme durations using stacked convolutional layers followed by self-attention.

    Uses FiLM conditioning: the speaker embedding is projected to scale (gamma) and shift (beta)
    and applied after LayerNorm at every layer.

    Self-attention after the conv stack lets each phoneme attend globally to all others,
    capturing sentence-level prosody (phrase-final lengthening, speech rate, stress) that
    the local convolutional receptive field cannot reach.
    """
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_layers=2, n_heads=2, spk_emb_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout
        self.spk_emb_dim = spk_emb_dim

        self.drop = nn.Dropout(p_dropout)
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # One gamma/beta pair per conv layer, plus one after attention norm, plus one after FFN norm.
        # Layout: [gamma_0, beta_0, gamma_1, beta_1, ..., gamma_attn, beta_attn, gamma_ffn, beta_ffn]
        self.n_film_steps = n_layers + 2
        self.spk_proj = nn.Linear(spk_emb_dim, self.n_film_steps * filter_channels * 2)
        # Initialize so FiLM is a no-op at the start of training: every gamma=1, every beta=0.
        nn.init.zeros_(self.spk_proj.weight)
        film_bias = self.spk_proj.bias.view(self.n_film_steps, 2, filter_channels)
        nn.init.ones_(film_bias[:, 0])   # gammas
        nn.init.zeros_(film_bias[:, 1])  # betas

        self.conv_layers.append(
            nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        )
        self.norm_layers.append(LayerNorm(filter_channels))

        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(filter_channels))

        # Self-attention over the full phoneme sequence, after local conv features are built.
        # filter_channels must be divisible by n_heads.
        self.attn = MultiHeadAttention(filter_channels, filter_channels, n_heads, p_dropout=p_dropout)
        self.attn_norm = LayerNorm(filter_channels)

        # Attention only routes a linear summary of context between phonemes. The FFN adds the
        # per-position nonlinear compute that lets each phoneme reason about that gathered context,
        # turning the attention step into a complete transformer block.
        self.ffn = FFN(filter_channels, filter_channels, filter_channels, kernel_size, p_dropout=p_dropout)
        self.ffn_norm = LayerNorm(filter_channels)

        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask, spk_emb):
        # Per-step FiLM params, shaped (batch, n_film_steps, 2, filter_channels, 1).
        # The 2 entries per step are gamma and beta.
        film = self.spk_proj(spk_emb).view(spk_emb.shape[0], self.n_film_steps, 2, self.filter_channels, 1)

        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            gamma, beta = film[:, i].unbind(dim=1)
            x = conv(x * x_mask)
            x = norm(x)
            x = nn.functional.silu(x)
            x = (x * gamma) + beta
            x = self.drop(x)

        # Self-attention with residual connection and layer norm.
        # The attention mask excludes padding positions from attention.
        attn_mask = (x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)).bool()
        x = self.attn_norm(x + self.attn(x * x_mask, attn_mask))
        # Re-inject FiLM: LayerNorm strips the per-channel speaker scale/shift, so without this the
        # speaker rhythm signal would be normalized away right before the duration projection.
        gamma_attn, beta_attn = film[:, -2].unbind(dim=1)
        x = (x * gamma_attn) + beta_attn

        # FFN sublayer: nonlinear per-position compute over the context attention just gathered.
        x = self.ffn_norm(x + self.ffn(x * x_mask, x_mask))
        gamma_ffn, beta_ffn = film[:, -1].unbind(dim=1)
        x = (x * gamma_ffn) + beta_ffn

        x = self.proj(x * x_mask)
        return x * x_mask



class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = nn.functional.silu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(nn.Module):
    """Transformer encoder with FiLM speaker conditioning.

    The speaker embedding is projected to a per-layer scale (gamma) and shift (beta) pair that is
    applied after each LayerNorm. This keeps the backbone speaker-agnostic at hidden_channels and
    re-injects the speaker character at every layer, instead of concatenating it into the input.
    """
    def __init__(
            self,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size=1,
            p_dropout=0.0,
            spk_emb_dim=96,
            **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_emb_dim = spk_emb_dim

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

        # One projection produces a gamma/beta pair for both LayerNorms in every layer.
        # Layout per layer: [gamma_1, beta_1, gamma_2, beta_2], each hidden_channels wide.
        self.film_params_per_layer = 4 * hidden_channels
        self.spk_proj = nn.Linear(spk_emb_dim, n_layers * self.film_params_per_layer)
        # Initialize so FiLM is a no-op at the start of training: every gamma=1, every beta=0.
        # Bias layout matches forward: per layer [gamma_1, beta_1, gamma_2, beta_2].
        nn.init.zeros_(self.spk_proj.weight)
        gamma_beta_per_layer = self.spk_proj.bias.view(n_layers, 4, hidden_channels)
        nn.init.ones_(gamma_beta_per_layer[:, 0::2])
        nn.init.zeros_(gamma_beta_per_layer[:, 1::2])

    def forward(self, x, x_mask, spk_emb):
        # Original code was not doing .bool()
        # scaled_dot_product_attention interprets the mask differently depending on its dtype:
        # bool tensor: False positions get -inf added before softmax, so they become zero after softmax — they are fully excluded from attention
        # float tensor: values are added directly as a bias to the attention logits. 
        # The mask ha 1.0 for valid and 0.0 for padding, originally, so padding positions were getting a get +0.0 bias, which has no effect.
        # That allowed padding to participate in attention.
        attn_mask = (x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)).bool()

        # Per-layer FiLM params, shaped (batch, n_layers, 4, hidden_channels, 1) so they broadcast
        # over time. The 4 entries per layer are gamma_1, beta_1, gamma_2, beta_2.
        film = self.spk_proj(spk_emb).view(spk_emb.shape[0], self.n_layers, 4, self.hidden_channels, 1)

        for i in range(self.n_layers):
            gamma_1, beta_1, gamma_2, beta_2 = film[:, i].unbind(dim=1)
            x = x * x_mask
            y = self.attn_layers[i](x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            x = (x * gamma_1) + beta_1
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
            x = (x * gamma_2) + beta_2
        x = x * x_mask
        return x


class TextEncoder(nn.Module):
    def __init__(
            self,
            encoder_params,
            duration_predictor_params,
            n_vocab,
            spk_emb_dim=128,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_feats = encoder_params.n_feats
        self.n_channels = encoder_params.n_channels
        self.spk_emb_dim = spk_emb_dim

        self.emb = nn.Embedding(n_vocab, self.n_channels)
        nn.init.normal_(self.emb.weight, 0.0, self.n_channels ** -0.5)

        self.prenet = ConvSiluNorm(
            self.n_channels,
            self.n_channels,
            self.n_channels,
            kernel_size=encoder_params.prenet_kernel_size,
            n_layers=6, # I have tested with 4 and it is considerably worse
            p_dropout=encoder_params.p_dropout,
        )

        self.encoder = Encoder(
            encoder_params.n_channels,
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
            self.spk_emb_dim,
        )
        
        self.proj_m = nn.Sequential(
            nn.Conv1d(self.n_channels, self.n_channels, 1),
            nn.SiLU(),
            nn.Conv1d(self.n_channels, self.n_feats, 1),
        )
        nn.init.xavier_uniform_(self.proj_m[2].weight)

        self.proj_w = DurationPredictor(
            encoder_params.n_channels,
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
            n_layers=duration_predictor_params.n_layers,
            n_heads=duration_predictor_params.n_heads,
            spk_emb_dim=self.spk_emb_dim,
        )

    def forward(self, x, x_lengths, speaker_embedding_enc, speaker_embedding_dur):
        """Run forward pass to the transformer based encoder and duration predictor

        Args:
            x (torch.Tensor): text input, a sequence of phoneme IDs, interleaved with separators,  
                as returned by multilingual_phonemizer()
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): text input lengths
                shape: (batch_size,)
            speaker_embedding (torch.Tensor): speaker embedding
                shape: (batch_size, spk_emb_dim)

        Returns:
            mu (torch.Tensor): A sequence with the mel frames predicted by the encoder, one per phoneme.
                shape: (batch_size, n_feats, max_text_length)
            logw (torch.Tensor): Log of phoneme durations predicted by the duration predictor.
                shape: (batch_size, 1, max_text_length)
            x_mask (torch.Tensor): mask for the text input
                shape: (batch_size, 1, max_text_length)
        """
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[2]), 1).to(x.dtype)

        x = self.prenet(x, x_mask)
        x = self.encoder(x, x_mask, speaker_embedding_enc)
        mu = self.proj_m(x) * x_mask
        
        # The author was feeding the encoder output into the Duration Predictor, undetached.   
        # But I thought the encoder output is too biased towards the acoustic meaning of the phonemes to make for 
        # a good input into duration prediction. I wanted to try feeding the raw phoneme embeddings instead, but we'd 
        # lose what the Encoder attention layers added to x, so I decided to add an attention and an FFN layer to the 
        # duration predictor. 
        # The predictor has 4 conv layers + attn + ffn + a final projection now.
        # After a lot of painful tests I found the duration predictor made some very egregious mistakes, albeit rare.
        # Speaker 6 has a very distinct pattern and the test sequence you see in debug.sh results in pronounced 
        # stuttering at the S in "time ssslipping" and the Fs in "fffar too fffrequent". Most of the durations are 
        # just fine, and the MCD improved, but I cannot get rid of those mistake even after 800 epochs.
        # As a result, I went back to using the Encoder output as input into Duration Prediction. That fixed the problem.
        # On the other hand, no matter what input I use, I must detach it.
        # I don't want the predictor pulling on anything Encoder related, because small prior changes destabilize 
        # MAS quickly. I have seen the prior loss spiking up, then never recovering because MAS tries to follow which
        # pulls prior further away which makes MAS follow and so on, never recovering.
        logw = self.proj_w(x.detach(), x_mask, speaker_embedding_dur)

        return mu, logw, x_mask
