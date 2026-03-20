# From Gemini Thinking

---

## ### 2. Padding Leakage (The Bias Problem)
In both `FFN` and `ConvReluNorm`, you apply masking like this:
```python
x = self.conv_1(x * x_mask)
x = torch.relu(x)
```
**The Problem:** Standard `nn.Conv1d` has a **bias** term by default. Even if the input `x` is zeroed out by the mask, the `conv_1` layer will add its bias to those padded positions. `ReLU` will then let any positive bias pass through.
When this reaches the next layer, your "padding" is no longer zeros—it's filled with constant bias values. This can "confuse" the encoder and lead to weird artifacts at the end of generated speech.

**The Fix:** Always apply the mask **after** the activation function, not just before the convolution.
```python
x = torch.relu(self.conv_1(x * x_mask)) * x_mask  # Mask again after ReLU
```

---

## ### 3. RoPE Buffer Management
Your `RotaryPositionalEmbeddings` class handles the cache like this:
```python
self.cos_cached = idx_theta2.cos()[:, None, None, :]
```
**The Problem:** You are assigning a new tensor to `self.cos_cached`. If you move your model to a GPU (`model.cuda()`) or convert it to half-precision (`model.half()`), this manually assigned tensor might not follow the rest of the model's state because it wasn't registered properly as a buffer during that specific assignment.

**The Fix:** Use `self.register_buffer` every time you update the cache, or ensure the cache is created on the correct device/dtype immediately by referencing `x.device` and `x.dtype`.

---

## ### 4. Numerical Stability in LayerNorm
Your custom `LayerNorm` is a manual implementation:
```python
variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
x = (x - mean) * torch.rsqrt(variance + self.eps)
```
**The Problem:** While mathematically correct, manual squared-difference variance is less numerically stable than the algorithms used in `torch.nn.LayerNorm` or `torch.nn.GroupNorm`. 
In FP16 training, this can occasionally lead to `NaN` gradients if the variance becomes extremely small.

**Recommendation:** Unless you have a specific architectural reason for this custom version, use:
```python
# Equivalent to your LayerNorm for (B, C, L)
self.norm = nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-5)
```

---

# From Claude Sonnet Extended 

Here's a thorough review organized by severity:

---

**8. Partial RoPE covers only half the head dimension**
```python
self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
```
`x_pass` (the non-rotated half) receives no positional information. 
This is a valid technique (used in some Llama variants), but if it's not intentional, 
the full head dimension `self.k_channels` should be used. 
A comment clarifying the choice would prevent future confusion.

---

## 🟢 Minor / Style

**9. `LayerNorm` normalizes over the channel axis (dim=1), not the last axis**
This is identical to the VITS/Matcha upstream implementation and is correct for `(B, C, T)` tensors — just worth noting it behaves 
like per-position instance normalization, not PyTorch's standard `nn.LayerNorm`.