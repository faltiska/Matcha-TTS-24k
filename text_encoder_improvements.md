There are three concrete improvements worth considering, ranging from a low-risk swap to a more structural change.

---

## 1. Pre-LN (Low risk, high stability payoff)

Your current Encoder uses **Post-LN** — normalize *after* the residual addition. The research consensus since GPT-2 era is that **Pre-LN** (normalize *before* the sublayer) trains more stably, is less sensitive to learning rate, and often converges faster — especially relevant since you said encoder quality is the bottleneck.

```python
def forward(self, x, x_mask):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    for i in range(self.n_layers):
        # Attention sublayer — Pre-LN
        residual = x
        x_normed = self.norm_layers_1[i](x) * x_mask
        y = self.attn_layers[i](x_normed, x_normed, attn_mask)
        y = self.drop(y)
        x = residual + y

        # FFN sublayer — Pre-LN
        residual = x
        y = self.ffn_layers[i](self.norm_layers_2[i](x), x_mask)
        y = self.drop(y)
        x = residual + y

    x = x * x_mask
    return x
```

---

## 2. SwiGLU FFN (Medium risk, meaningful expressivity gain)

Your FFN uses ReLU. **SwiGLU** (`x * SiLU(gate)`) has empirically outperformed ReLU in virtually every transformer that has tried it (LLaMA, PaLM, etc.). The trade-off: the first conv projects to `2 × filter_channels` (to produce both value and gate), so it costs ~33% more parameters in the FFN, but is usually net positive.

```python
class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        # Project to 2x to get value + gate
        self.conv_1 = nn.Conv1d(in_channels, filter_channels * 2, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x, gate = x.chunk(2, dim=1)
        x = x * torch.nn.functional.silu(gate)  # SwiGLU
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask
```

---

## 3. Conformer convolution module (Higher risk, highest potential upside for TTS)

This is the one I'd be most excited about for your use case. The **Conformer** (Gulati et al., 2020) was designed specifically for speech: it adds a lightweight depthwise convolution sublayer between attention and FFN. Attention captures long-range phoneme dependencies; depthwise conv captures local co-articulation patterns (adjacent phonemes bleeding into each other). Many modern TTS systems (NaturalSpeech 2, Voicebox, etc.) have adopted exactly this architecture.

```python
class ConformerConvModule(nn.Module):
    """Conformer-style convolution sublayer.

    Pointwise expansion with GLU gating → depthwise conv → norm → SiLU → pointwise projection.
    Captures local phonetic context that attention misses.
    """
    def __init__(self, channels, kernel_size=31, p_dropout=0.0):
        super().__init__()
        # Expand + GLU gate
        self.pointwise_conv1 = nn.Conv1d(channels, channels * 2, 1)
        # Depthwise (each channel convolves independently — cheap, local)
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels
        )
        self.norm = LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.pointwise_conv1(x * x_mask)
        x, gate = x.chunk(2, dim=1)
        x = x * torch.sigmoid(gate)         # GLU gate
        x = self.depthwise_conv(x * x_mask)
        x = self.norm(x)
        x = torch.nn.functional.silu(x)
        x = self.pointwise_conv2(x)
        x = self.drop(x)
        return x * x_mask
```

Then the Encoder becomes:

```python
class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        conv_kernel_size=31,   # <-- new: kernel for depthwise conv
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.conv_modules = nn.ModuleList()      # <-- new
        self.norm_layers_conv = nn.ModuleList()  # <-- new
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout)
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.conv_modules.append(ConformerConvModule(hidden_channels, conv_kernel_size, p_dropout))
            self.norm_layers_conv.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            # 1. Self-attention sublayer (Pre-LN)
            residual = x
            x_normed = self.norm_layers_1[i](x) * x_mask
            y = self.attn_layers[i](x_normed, x_normed, attn_mask)
            y = self.drop(y)
            x = residual + y

            # 2. Depthwise conv sublayer (Pre-LN) — local phonetic context
            residual = x
            y = self.conv_modules[i](self.norm_layers_conv[i](x), x_mask)
            x = residual + y

            # 3. FFN sublayer (Pre-LN)
            residual = x
            y = self.ffn_layers[i](self.norm_layers_2[i](x), x_mask)
            y = self.drop(y)
            x = residual + y

        x = x * x_mask
        return x
```

---

## Summary

| Change | Risk | Expected gain |
|---|---|---|
| Pre-LN | Very low — just reorder operations | More stable training, less LR sensitivity |
| SwiGLU FFN | Low — swap activation, slight param increase | Better encoder representations |
| Conformer conv module | Medium — new sublayer, new hyperparameter `conv_kernel_size` | Better local phoneme modeling → better duration prediction and cleaner mel output |

One practical note: `conv_kernel_size=31` is the standard Conformer value. Since your sequences are phoneme-level (not frame-level), you could experiment with smaller values like 15 or even 9 — the "local context" window means something different at phoneme granularity than at 10ms acoustic frames.