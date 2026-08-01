from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.models.attention import (
    AdaLayerNorm,
    AdaLayerNormZero,
)
from diffusers.models.attention_processor import Attention
from diffusers.utils.torch_utils import maybe_allow_in_graph


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    It fuses the feed-forward input projection into the activation, so the periodic parameters are
    per output channel and the activation runs on the last dimension.
    Shape:
        - Input: (B, T, in_features)
        - Output: (B, T, out_features)
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        a1 = SnakeBeta(256, 1024)
        x = torch.randn(1, 100, 256)
        x = a1(x)
    """

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = nn.Linear(in_features, out_features)

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(self.in_features))
            self.beta = nn.Parameter(torch.zeros(self.in_features))
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta

        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        act_fn = SnakeBeta(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class TimestepConditionedTransformerBlock(nn.Module):
    """
    Transformer block (self-attention + feed-forward) where both sub-layers are conditioned on
    the diffusion timestep embedding via Adaptive Layer Norm Zero (adaLN-Zero).

    A single adaLN-Zero projection (norm1) produces all 6 modulation parameters from the
    timestep embedding:
      - shift and scale applied to the hidden states before attention (inside norm1)
      - a gate that scales the attention output before the residual add
      - shift, scale, and gate for the feed-forward sub-layer, applied on top of norm3,
        which is a plain non-affine layer norm

    Both gates are initialised to zero by zero_initialize_timestep_modulation, which the
    decoder calls after its global weight initialisation. The block therefore starts as an
    identity function and the timestep conditioning is gradually activated during training.

    Args:
        dim: Hidden dimension (must equal num_attention_heads * attention_head_dim).
        num_attention_heads: Number of self-attention heads.
        attention_head_dim: Dimension per attention head.
        time_embed_dim: Dimension of the pre-computed timestep embedding fed in as `timestep`.
        dropout: Dropout probability applied in attention and feed-forward.
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, time_embed_dim: int, dropout: float = 0.0):
        super().__init__()

        # adaLN-Zero norm for the self-attention sub-layer.
        # num_embeddings=None means it expects a pre-computed embedding via the `emb` kwarg,
        # rather than looking up a discrete timestep index.
        # It also produces the shift, scale and gate used by the feed-forward sub-layer below.
        self.norm1 = AdaLayerNormZero(embedding_dim=dim, num_embeddings=None)

        # Projects the timestep embedding into the dimension adaLN-Zero expects.
        self.time_proj = nn.Linear(time_embed_dim, dim)

        self.attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
        )

        # Plain norm for the feed-forward sub-layer. It carries no learnable affine of its own
        # because the shift and scale are supplied by norm1's modulation parameters.
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.ff = FeedForward(dim, dropout=dropout)

    def zero_initialize_timestep_modulation(self):
        """
        Restores adaLN-Zero behaviour after the decoder's global weight initialisation,
        which applies Xavier initialisation to every linear layer including this one.
        Zeroing the modulation projection makes both gates start at zero, so the block
        starts as an identity function.
        """
        nn.init.zeros_(self.norm1.linear.weight)
        nn.init.zeros_(self.norm1.linear.bias)

    def forward(self, hidden_states: torch.FloatTensor, attention_mask=None, timestep=None, **kwargs):
        # Project timestep embedding to match the block's hidden dimension.
        time_emb = self.time_proj(timestep)

        # The decoder supplies a 0.0 / 1.0 float mask, which scaled dot product attention would treat
        # as values to add to the attention logits, leaving padded frames visible to the valid ones.
        # A boolean mask selects the path where padded keys are genuinely excluded.
        if attention_mask is not None:
            attention_mask = attention_mask.bool()

        # --- Self-attention sub-layer ---
        # norm1 returns normalised states + 4 modulation tensors (gate_msa used here, rest for FFN)
        normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=time_emb)
        attn_output = self.attn(normed, attention_mask=attention_mask)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # --- Feed-forward sub-layer ---
        normed = self.norm3(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(normed)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        return hidden_states
