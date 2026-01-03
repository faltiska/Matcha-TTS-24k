# Matcha-TTS Architecture and Components

This document summarizes the core architecture and synthesis/training pipeline of Matcha-TTS.

## High-level
- Non-autoregressive TTS that generates mel-spectrograms using Conditional Flow Matching (CFM), solved with an ODE integration.
- Two learnable stages:
  1) Text encoder with duration prediction and monotonic alignment
  2) Conditional flow decoder that maps noise to mel conditioned on encoder features
- A separate vocoder (e.g., HiFi-GAN, Vocos) converts the generated mel to waveform.

## Core Components

### 1) TextEncoder
Files: `matcha/models/components/text_encoder.py`
- Token embedding (+ optional convolutional prenet)
- Transformer-style encoder stack with:
  - Multi-Head Attention
  - Rotary Positional Embeddings (RoPE)
  - LayerNorm + FFN blocks
- Outputs:
  - mu (prior mel-like features per text frame) via `proj_m`
  - logw (log token durations) via `DurationPredictor`
- Multi-speaker support: if `n_spks > 1`, a learned speaker embedding is concatenated as extra channels.

### 2) Alignment and Length Expansion
Files: `matcha/models/matcha_tts.py`, `matcha/utils/monotonic_align`
- Converts predicted durations exp(logw) into an alignment path `attn`:
  - In inference: use predicted durations to build path
  - In training: Monotonic Alignment Search (MAS) over a Gaussian prior around `mu` to obtain supervision
- Expands text-frame `mu` to mel-frame `mu_y` using the alignment `attn`
- `length_scale` adjusts speaking rate by scaling durations.

### 3) Conditional Flow Matching (CFM) Decoder
Files: `matcha/models/components/flow_matching.py`, `matcha/models/components/decoder.py`
- Establishes an ODE from noise to target mel conditioned on `mu_y` and masks
- Estimator network: a UNet-like `Decoder` that predicts the conditional flow field
- Inference:
  - Start from noise `z`
  - Integrate dx/dt = estimator(x, t, cond) over t ∈ [0, 1] using the torchdiff built-in ODE implementation which offers multiple methods
  - Solver method is specified in `cfm_params.solver`
  - `n_timesteps` controls the number of integration steps
  - `temperature` scales the terminal noise
- Multi-speaker conditioning is passed when applicable.

## Training Objectives
Files: `matcha/models/matcha_tts.py`
- Duration loss: between predicted `logw` and MAS-derived durations
- Prior loss: encourages target mel `y` to match `N(mu_y, I)` (i.e., Gaussian around encoder-expanded prior)
- Flow matching loss: MSE between the estimator’s vector field and the true conditional flow on randomly sampled `t`

## Inference Pipeline
1) Text → phoneme IDs → TextEncoder → `mu` (text-time), `logw`
2) Durations → alignment path `attn` → expand to `mu_y` (mel-time)
3) CFM decoder integrates from noise to mel using the ODE solver
4) Denormalize mel → external vocoder produces waveform

## Key Controls
- `solver`: ODE integration method (configurable via `cfm_params.solver`)
- `n_timesteps`: speed/quality trade-off for ODE integration
- `temperature`: output stochasticity
- `length_scale`: speaking rate control
- Speaker ID (if multi-speaker): selects speaker embedding

## Shapes (typical)
- Text `x`: (B, T_text), lengths `x_lengths`
- Encoder `mu`: (B, n_feats, T_text)
- Expanded `mu_y` and decoder output mel: (B, n_feats, T_mel)
- Masks: `x_mask` (B, 1, T_text), `y_mask` (B, 1, T_mel)
- Speaker embedding (if used): learned embedding concatenated as channels
