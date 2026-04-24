# Style Encoder

## Purpose

The Style Encoder lets you add a new voice to a trained Matcha-TTS model using a small set of recordings (~50 Harvard Sentences), without retraining the full model.

Matcha-TTS represents each speaker as a single learned vector of numbers:
- The **speaker embedding** (96 numbers) — tells the text encoder what this speaker's voice sounds like, shaping how phonemes are mapped to mel frames, and also drives the duration predictor

For a known speaker, this vector comes from a lookup table trained over 4+ days. The Style Encoder learns to predict it directly from a short audio recording, so a new speaker does not need to go through full training.

---

## Architecture

### AcousticStyleEncoder (ASE)

Predicts the **speaker embedding** from a mel spectrogram.

```
mel spectrogram  (100 frequency bins × T time frames)
  → 4 convolutional layers (kernel=5, 256 channels) + ReLU, masked
  → average over time (ignoring padding)
  → linear projection → 96 numbers
```

The ASE reads the mel, runs it through a stack of convolutions to extract voice characteristics, averages everything into a single summary vector, then projects it down to the 96 numbers that Matcha's encoder and duration predictor both expect.

---

## Training

The Matcha checkpoint is frozen — none of its weights change. Only the ASE weights are trained.

### What happens for each batch

```
1. Run the frozen Matcha encoder with the real speaker embedding from the lookup table
   → mu_x_real : the encoder's phoneme-to-mel output using the real speaker embedding

2. ASE reads the mel spectrogram → predicts the speaker embedding (96 numbers)

3. Run the frozen Matcha encoder with the predicted embedding → mu_x_pred

4. Loss: mean squared error between mu_x_pred and mu_x_real
```

The loss teaches the ASE to predict a speaker embedding that makes the encoder produce the same phoneme-to-mel mapping as it would with the real embedding.

---

## Files

| File | Description |
|------|-------------|
| `matcha/models/style_encoder.py` | `AcousticStyleEncoder`, `StyleEncoderLightningModule` |
| `matcha/train_style_encoder.py` | Training entry point (mirrors `train.py`) |
| `configs/model/style_encoder/default.yaml` | Model hyperparameters |
| `configs/train_style_encoder.yaml` | Top-level Hydra config |

---

## Configuration

`configs/model/style_encoder/default.yaml`:

```yaml
matcha_checkpoint_path: ???   # required — path to trained Matcha .ckpt

n_feats: 100                  # must match Matcha data config
spk_emb_dim: 96               # must match Matcha spk_emb_dim_enc

ase_hidden_channels: 256
ase_n_layers: 4
```

All dimension values must match the Matcha checkpoint exactly.

---

## Running training

```bash
python -m matcha.train_style_encoder
```

---

## Adding a new speaker

### 1. Prepare the corpus

Prepare a CSV and precompute mels as you would for training (see README):
```bash
python -m matcha.utils.precompute_mels -i configs/data/extra-speakers-24k.yaml
```

### 2. Run add_speaker

```bash
python -m matcha.add_speaker \
  --style-encoder-ckpt logs/train_style_encoder/style_encoder_v2/runs/2026-04-23_21-18-27/checkpoints/checkpoint_epoch=249.ckpt \
  --matcha-ckpt logs/train/v17/checkpoint_epoch=603.ckpt \
  --csv data/extra-speakers-24k/train.csv \
  --output checkpoint_with_new_speaker.ckpt
```

The script prints the new speaker's ID at the end.

### 3. Synthesise

```bash
python -m matcha.cli \
  --checkpoint_path checkpoint_with_new_speaker.ckpt \
  --spk 10 \
  --text "Bună, numele meu este Daria."
```

---

## Inference internals

At inference time, the lookup table call is replaced with the ASE:

```python
pred_spk_emb = acoustic_style_encoder(mel, mel_mask)
```

The predicted embedding is passed to `matcha.encoder(...)` for both the encoder and duration predictor, in place of the lookup table output.
