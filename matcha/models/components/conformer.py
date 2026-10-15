"""
Timestep-conditioned Conformer block for the flow matching Decoder.

Replaces the third party `conformer` package (lucidrains, 0.1.0), which could not be used here as it was:
 - it ignored the diffusion timestep,
 - its convolution module used an unmasked BatchNorm1d, whose statistics mix in padded frames, depend on the
   batch, differ between training and inference, and get averaged over all timesteps,
 - its depthwise convolution read the padded frames,
 - its attention was a hand-written einsum with Shaw relative positions clamped at 512 frames, no SDPA kernels.

Design, per block (macaron layout from the Conformer paper, arXiv:2005.08100):

    x = x + g1 * FFN (modulate(LN(x), shift1, scale1))
    x = x + g2 * MHSA(modulate(LN(x), shift2, scale2))
    x = x + g3 * Conv(modulate(LN(x), shift3, scale3))
    x = x + g4 * FFN (modulate(LN(x), shift4, scale4))

 - Timestep conditioning: adaLN-Zero (DiT, arXiv:2212.09748) on every sub-layer. One SiLU + Linear produces the 12
   shift / scale / gate vectors. It is zero initialised, so every gate starts at zero and the block starts as an
   identity on the valid frames. The paper's 0.5 macaron weight on the two FFNs is absorbed by the learned gates.
 - Attention: fused QKV, RMS QK-norm (keeps the logits bounded under bf16, as in SD3 / Flux), rotary positions on
   half of each head (same convention as the text encoder, computed on the fly so there is no length limit), and
   PyTorch SDPA with a boolean key mask, so padded keys are excluded rather than down-weighted.
 - Convolution module: pointwise -> GLU -> mask -> depthwise -> LayerNorm -> SiLU -> pointwise. The input is
   zeroed on padded frames before the depthwise convolution, so a valid frame next to the padding sees exactly
   what it would see at the end of an unpadded sequence. LayerNorm is per frame, so nothing in the block computes
   statistics across time or across the batch.
 - Feed-forward: the project's SnakeBeta FeedForward, with a final dropout as in the Conformer paper.
 - No final LayerNorm: the residual stream goes straight into the next ResnetBlock1D, which normalises it with
   MaskedGroupNorm.

Valid frames of the output do not depend on how much masked padding follows them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from matcha.models.components.transformer import FeedForward


class RotaryEmbedding(nn.Module):
    """Rotary position embedding applied to the first `rotary_fraction` of each head's dimensions.

    The angles are computed for the actual sequence length on every call, so there is no cache to outgrow.
    Positions start at 0 for every sequence: padding is always at the end, so it does not shift them.
    """

    def __init__(self, head_dim: int, rotary_fraction: float = 0.5, base: float = 10_000.0):
        super().__init__()
        rotary_dim = int(head_dim * rotary_fraction)
        rotary_dim -= rotary_dim % 2
        assert rotary_dim > 0, "rotary_fraction leaves no dimensions to rotate"
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate(self, x, cos, sin):
        x_rot, x_pass = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return torch.cat((rotated, x_pass), dim=-1)

    def forward(self, q, k):
        """q, k: (batch, heads, time, head_dim), in fp32. Returns them rotated, still in fp32."""
        positions = torch.arange(q.shape[2], device=q.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)
        cos, sin = angles.cos(), angles.sin()
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


class ConformerSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, dropout: float = 0.0, rotary_fraction: float = 0.5):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim

        self.qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.rotary = RotaryEmbedding(head_dim, rotary_fraction)
        self.out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        # Initialise q, k and v separately, so the xavier fan-out matches a single projection and not all three.
        inner_dim = self.heads * self.head_dim
        for start in range(0, 3 * inner_dim, inner_dim):
            nn.init.xavier_uniform_(self.qkv.weight[start : start + inner_dim])

    def forward(self, x, key_mask):
        """
        Args:
            x: (batch, time, dim)
            key_mask: bool, True for valid frames, (batch, time)
        """
        q, k, v = rearrange(self.qkv(x), "b t (three h d) -> three b h t d", three=3, h=self.heads)

        # QK-norm and RoPE in fp32, then back to the compute dtype that SDPA needs for all three inputs.
        q, k = self.rotary(self.q_norm(q.float()), self.k_norm(k.float()))
        q, k = q.to(v.dtype), k.to(v.dtype)

        # Key mask only: every query row keeps at least one valid key, so no row is fully masked.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.dropout(self.out(out))


class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        assert kernel_size % 2 == 1, "conv_kernel_size must be odd so the convolution stays centred"
        self.pointwise_in = nn.Linear(dim, 2 * dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pointwise_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, frame_mask):
        """
        Args:
            x: (batch, time, dim)
            frame_mask: 1.0 for valid frames, (batch, time, 1), in x's dtype
        """
        x = F.glu(self.pointwise_in(x), dim=-1)
        # Zero the padding so the depthwise kernel sees the same zeros as at an unpadded sequence end.
        x = rearrange(x * frame_mask, "b t c -> b c t")
        x = rearrange(self.depthwise(x), "b c t -> b t c")
        x = F.silu(self.norm(x))
        return self.dropout(self.pointwise_out(x))


class TimestepConditionedConformerBlock(nn.Module):
    """
    Macaron Conformer block (FFN, self-attention, convolution, FFN) with adaLN-Zero timestep conditioning on every
    sub-layer. See the module docstring for the design.

    Args:
        dim: Hidden dimension.
        num_attention_heads: Number of self-attention heads.
        attention_head_dim: Dimension per attention head.
        time_embed_dim: Dimension of the timestep embedding fed in as `timestep`.
        dropout: Applied inside and at the output of both FFNs, and at the output of attention and convolution.
        ff_mult: Hidden size multiplier of each of the two FFNs.
        conv_kernel_size: Depthwise convolution kernel, odd. 31 frames is 330 ms at the top UNet level.
        rotary_fraction: Fraction of each head's dimensions that receive rotary positions.
    """

    NUM_SUBLAYERS = 4

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        ff_mult: int = 2,
        conv_kernel_size: int = 31,
        rotary_fraction: float = 0.5,
    ):
        super().__init__()
        self.dim = dim

        # shift, scale and gate for each sub-layer.
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 3 * self.NUM_SUBLAYERS * dim))

        self.ff1 = FeedForward(dim, mult=ff_mult, dropout=dropout, final_dropout=True)
        self.attn = ConformerSelfAttention(dim, num_attention_heads, attention_head_dim, dropout, rotary_fraction)
        self.conv = ConformerConvModule(dim, conv_kernel_size, dropout)
        self.ff2 = FeedForward(dim, mult=ff_mult, dropout=dropout, final_dropout=True)

        self.reset_parameters()

    def reset_parameters(self):
        """Block-specific initialisation. Decoder.initialize_weights() calls it again after its generic pass,
        which would otherwise overwrite the zero adaLN projection with xavier."""
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)
        self.attn.reset_parameters()
        # Xavier on a depthwise kernel counts all output channels as fan-out (bound ~0.02 for dim 384, kernel 31),
        # far below the PyTorch default for grouped convolutions (~0.18). Restore the default.
        self.conv.depthwise.reset_parameters()

    def _modulate(self, x, shift, scale):
        return F.layer_norm(x, (self.dim,), eps=1e-6) * (1 + scale) + shift

    def forward(self, hidden_states, attention_mask, timestep, **kwargs):
        """
        Args:
            hidden_states: (batch, time, dim)
            attention_mask: 1 for valid frames, 0 for padding, (batch, time), float or bool
            timestep: timestep embedding, (batch, time_embed_dim)
        Returns:
            (batch, time, dim), zero on padded frames
        """
        key_mask = attention_mask.bool()
        frame_mask = key_mask.unsqueeze(-1).to(hidden_states.dtype)

        modulation = self.adaln(timestep).unsqueeze(1).chunk(3 * self.NUM_SUBLAYERS, dim=-1)
        (shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3, shift4, scale4, gate4) = modulation

        x = hidden_states
        x = x + gate1 * self.ff1(self._modulate(x, shift1, scale1))
        x = x + gate2 * self.attn(self._modulate(x, shift2, scale2), key_mask)
        x = x + gate3 * self.conv(self._modulate(x, shift3, scale3), frame_mask)
        x = x + gate4 * self.ff2(self._modulate(x, shift4, scale4))
        return x * frame_mask
