"""
Shared attention primitives used by both the text encoder and the conformer duration predictor.
Kept in a separate module to avoid circular imports.
"""
import torch
import torch.nn as nn
from einops import rearrange


class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    For each phoneme, treats consecutive pairs of values in its embedding as 2D vectors
    and rotates them by an angle proportional to the phoneme's position.
    This makes attention dot products sensitive to the relative distance between phonemes
    rather than their absolute positions.
    """

    def __init__(self, d: int, base: int = 10_000):
        """
        * `d` is the number of embedding values to apply RoPE to (half the per-head embedding size)
        * `base` is the constant used for calculating Theta
        """
        super().__init__()

        self.base = base
        self.d = int(d)

        # The total phonetic representation, including annotations and separators must be shorter than this.
        # The server should enforce a max text length that can fit into this. The client app too.
        # With the (pre, phoneme, post) tokenization scheme, a 1000 symbols input text will be less than 3000 symbols 
        # long after tokenization, but since some symbols are multibyte, I want some extra space, just inh case.
        self.max_seq_len = 4000
        # Pre-allocate and fill cos/sin caches
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d))
        seq_idx = torch.arange(self.max_seq_len).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.register_buffer('cos_cached', idx_theta2.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', idx_theta2.sin()[None, None, :, :], persistent=False)

    def _neg_half(self, x: torch.Tensor):
        # Rearranges x so the second half of the values comes first, negated:
        # [-x[d/2:], x[:d/2]]
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[batch_size, n_heads, seq_len, d]`
        """
        seq_len = x.shape[2]
        assert seq_len <= self.max_seq_len, f"Phonetic representation too long, exceeds RoPE cache size {self.max_seq_len}"

        # Split the embedding values: RoPE is applied only to the first d values, the rest are passed through unchanged.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[:, :, :seq_len]) + (neg_half_x * self.sin_cached[:, :, :seq_len])

        return torch.cat((x_rope, x_pass), dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            channels,
            out_channels,
            n_heads,
            p_dropout=0.0,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.k_channels = channels // n_heads
        # Fused query/key/value projection: one 1x1 conv produces all three, halving GEMM launches.
        self.conv_qkv = torch.nn.Conv1d(channels, channels * 3, 1)

        # from https://nn.labml.ai/transformers/rope/index.html
        self.rope = RotaryPositionalEmbeddings(self.k_channels * 0.5)

        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)

        # Init each q/k/v slice independently so xavier fan-in matches a per-projection conv.
        for slice_start in range(0, channels * 3, channels):
            torch.nn.init.xavier_uniform_(self.conv_qkv.weight[slice_start:slice_start + channels])

    def forward(self, x, attn_mask=None):
        q, k, v = self.conv_qkv(x).chunk(3, dim=1)

        x = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        query = self.rope(query)
        key = self.rope(key)

        attn_mask = mask.bool() if mask is not None else None
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=self.p_dropout if self.training else 0.0,
        )
        output = rearrange(output, "b h t c -> b (h c) t")
        return output
