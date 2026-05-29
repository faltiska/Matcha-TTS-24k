# Style Encoder

## Purpose

The Style Encoder lets you add a new voice to a trained Matcha-TTS model using a small set of recordings (~50 Harvard Sentences), without retraining the full model.

Matcha-TTS represents each speaker as two learned vectors of numbers (96 each):
- One drives the text encoder, shaping how phonemes are mapped to mel frames
- One drives the duration predictor

For a known speaker, these vectors come from two lookup tables trained over 4+ days. The Style Encoder learns to predict them directly from a short audio recording, so a new speaker does not need to go through full training.

---

## Architecture

### StyleEncoder

Predicts both speaker embeddings from a mel spectrogram.

```
mel spectrogram  (100 frequency bins × T time frames)
  → 4 convolutional layers (kernel=5, 256 channels) + ReLU, masked
  → average over time (ignoring padding)
  → two linear projections → 96 numbers each
        • encoder embedding   (for the text encoder)
        • duration embedding  (for the duration predictor)
```

The conv stack extracts voice characteristics, averages everything into a single summary vector, then two linear heads project it into the two 96-number embeddings that Matcha's encoder and duration predictor expect.

---

## Training

The Matcha checkpoint is frozen — none of its weights change. Only the StyleEncoder weights are trained.

### What happens for each batch

```
1. StyleEncoder reads the mel spectrogram → predicts both speaker embeddings

2. Run the frozen Matcha encoder with the real embeddings from the lookup tables
   → mu_x_real (phoneme-to-mel output) and logw_real (predicted log-durations)

3. Run the frozen Matcha encoder with the predicted embeddings
   → mu_x_pred and logw_pred

4. Acoustic loss: smooth L1 between mu_x_pred and mu_x_real
   Rhythm loss:   smooth L1 between logw_pred and logw_real
   Total loss:    acoustic + rhythm
```

The two losses teach the StyleEncoder to predict embeddings that make the encoder reproduce both the phoneme-to-mel mapping (acoustic) and the predicted durations (rhythm) it would have produced with the real embeddings.

---

## Files

| File | Description |
|------|-------------|
| `matcha/models/style_encoder.py` | `StyleEncoder`, `StyleEncoderLightningModule` |
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

Prepare the corpus as you would for training:
```bash
./prepare_corpus.sh configs/data/extra-speakers-24k.yaml
```

### 2. Run add_speaker

```bash
python -m matcha.add_speaker \
  --style-encoder-ckpt logs/train_style_encoder/style_encoder_v2/runs/2026-05-24_12-49-14/checkpoints/checkpoint_epoch=599.ckpt \
  --matcha-ckpt logs/train/v19-prod/checkpoint_epoch=1281.ckpt \
  --csv data/extra-speakers-24k/train.csv \
  --output checkpoint_with_new_speaker.ckpt
```

Recordings are processed one at a time. For each one, the StyleEncoder predicts a pair of embeddings; the two predictions are then averaged independently across all recordings and written into the expanded checkpoint.

The script prints the new speaker's ID at the end.

### 3. Synthesise

```bash
python -m matcha.cli \
  --checkpoint_path checkpoint_with_new_speaker.ckpt \
  --spk 15 \
  --text "Bună, numele meu este Daria."
```

---

## Inference internals

At inference time, the lookup table calls are replaced with the StyleEncoder:

```python
pred_emb_enc, pred_emb_dur = style_encoder(mel, mel_mask)
```

The two predicted embeddings are passed to `matcha.encoder(...)` in place of the lookup table outputs — one for the text encoder, one for the duration predictor.


## Improvement ideas

Weakness 1 — pooling throws away everything but the mean
masked_mean_pool collapses the time axis to a single average. Two cheap, well-established upgrades:
•
Stats pooling (mean + std): concatenate the masked mean and the masked standard deviation over time. Doubles the pooled width (so the linear heads' input goes hidden → 2*hidden), captures variability, and is what x-vector speaker encoders use. Minimal code, no new layers.
•
Attentive stats pooling: learn per-frame weights, then take weighted mean+std. Strictly more expressive (ECAPA-TDNN uses it), but adds a small attention layer. More moving parts.
For minimal change with a real gain, I'd start with mean+std stats pooling.
Weakness 2 — shallow conv stack, no normalization
The current stack is 4 plain Conv1d+ReLU, ~17-frame receptive field, no normalization, no residuals. Two issues: receptive field too short to capture speaker character across longer spans, and no normalization makes a deeper stack harder to train. Fixes, in order of effort:
•
Add normalization + residuals to the existing conv blocks so you can deepen safely (your text_encoder.py already has LayerNorm and ConvSiluNorm — same idea, can mirror that style).
•
Widen the receptive field, either by going deeper or by dilations.
The honest caveat I gave before still holds: these two fix timbre fidelity, which is already your good axis. They won't fix the accent — that's the entanglement/main-model story. So I'd treat them as a second, lower-priority workstream, not the headline.
That leaves us with potentially three coordinated changes:
1.
Main model: concat → FiLM/AdaLN in the Encoder (fixes the accent; the headline).
2.
Style Encoder pooling: mean → mean+std.
3.
Style Encoder backbone: add norm/residuals, widen receptive field.
Before I write a plan I still need the one decision from earlier — shared FiLM vs per-layer AdaLN in the Encoder — and now also: do you want all three in one plan, or just the main-model change first (since that's the one that targets your actual complaint)? My recommendation: per-layer AdaLN, and do change #1 first on its own so you can confirm the accent improves before spending effort on #2/#3.