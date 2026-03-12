# Style Encoder

## Purpose

The Style Encoder lets you add a new voice to a trained Matcha-TTS model using a small set of recordings (~50 Harvard Sentences), without retraining the full model.

Matcha-TTS represents each speaker as two learned vectors of numbers:
- The **encoder voice embedding** (96 numbers) — tells the text encoder what this speaker's voice sounds like, shaping how phonemes are mapped to mel frames
- The **duration voice embedding** (64 numbers) — tells the duration predictor how long this speaker holds each sound

For a known speaker, these vectors come from lookup tables that were trained over 4+ days. The Style Encoder learns to predict both vectors directly from a short audio recording, so a new speaker does not need to go through full training.

---

## Architecture

### AcousticStyleEncoder (ASE)

Predicts the **encoder voice embedding** from a mel spectrogram.

```
mel spectrogram  (100 frequency bins × T time frames)
  → 4 convolutional layers (kernel=5, 256 channels) + ReLU, masked
  → average over time (ignoring padding)
  → linear projection → 96 numbers
```

The ASE reads the mel, runs it through a stack of convolutions to extract voice characteristics, averages everything into a single summary vector, then projects it down to the 96 numbers that Matcha's encoder expects.

### RhythmStyleEncoder (RSE)

Predicts the **duration voice embedding** from a single input:

- **ASE output** (96 numbers) — the voice identity vector produced by the ASE above

```
ASE output  (96 numbers)
  → linear projection → 64 numbers
```

The ASE output already encodes who is speaking. The RSE is a single linear layer that projects that identity vector into the duration embedding space.

---

## Training

The Matcha checkpoint is frozen — none of its weights change. Only the ASE and RSE weights are trained.

### What happens for each batch

```
1. Run the frozen Matcha encoder with the real voice embeddings from the lookup table
   → mu_x_real : the encoder's phoneme-to-mel output using the real voice embedding
   → logw_mas  : the MAS ground-truth durations (how long each phoneme lasted in the real audio)

2. ASE reads the mel spectrogram → predicts the encoder voice embedding (96 numbers)

3. RSE takes the ASE output → predicts the duration voice embedding (64 numbers)

4. Run the frozen Matcha encoder twice, once per loss, to keep gradients isolated:
   - With pred_dur_emb detached → mu_x_pred (ASE loss cannot reach RSE weights)
   - With pred_enc_emb detached → logw_pred  (RSE loss cannot reach ASE weights)

5. Losses:
   ASE loss = mean squared error between mu_x_pred and mu_x_real
   RSE loss = mean squared error between logw_pred and logw_mas
   total    = ASE loss + RSE loss
```

The ASE loss teaches the ASE to predict a voice embedding that makes the encoder produce the same phoneme-to-mel mapping as it would with the real embedding. The RSE loss teaches the RSE to predict a duration embedding that makes the duration predictor reproduce the real MAS durations.

### Why there are two second Matcha forward passes, and why neither is wrapped in `torch.no_grad()`

Each loss must train only its own encoder:
- The ASE loss pass detaches `pred_dur_emb`, so the gradient from `ase_loss` cannot reach RSE weights.
- The RSE loss pass detaches `pred_enc_emb`, so the gradient from `rse_loss` cannot reach ASE weights.

Neither pass is wrapped in `torch.no_grad()` because the gradient must travel back through the frozen Matcha encoder to reach the predicted embeddings, and from there into ASE or RSE weights. Matcha's own weights are frozen via `requires_grad = False`, so they receive the gradient signal but do not update.

### Why the phoneme content vectors are detached

The phoneme content vectors (`x_dp`) are produced by the frozen Matcha encoder. We pass them to RSE as a fixed input — we do not want the gradient from the RSE loss to flow back into the frozen encoder through this path. Detaching them cuts that connection.

---

## Files

| File | Description |
|------|-------------|
| `matcha/models/style_encoder.py` | `AcousticStyleEncoder`, `RhythmStyleEncoder`, `StyleEncoderLightningModule` |
| `matcha/train_style_encoder.py` | Training entry point (mirrors `train.py`) |
| `configs/model/style_encoder/default.yaml` | Model hyperparameters |
| `configs/train_style_encoder.yaml` | Top-level Hydra config |

---

## Configuration

`configs/model/style_encoder/default.yaml`:

```yaml
matcha_checkpoint_path: ???   # required — path to trained Matcha .ckpt

n_feats: 100                  # must match Matcha data config
n_channels: 192               # must match Matcha encoder_params.n_channels
spk_emb_dim_enc: 96           # must match Matcha spk_emb_dim_enc
spk_emb_dim_dur: 64           # must match Matcha spk_emb_dim_dur

ase_hidden_channels: 256
ase_n_layers: 4
```

All dimension values must match the Matcha checkpoint exactly.

---

## Running training

```bash
python -m matcha.train_style_encoder \
  model.matcha_checkpoint_path=logs/train/.../checkpoint.ckpt
```

---

## Adding a new speaker

### 1. Prepare the corpus

Prepare a CSV and precompute mels as you would for training (see README):
```bash
python -m matcha.utils.precompute_corpus -i configs/data/extra-speakers-24k.yaml
```

### 2. Run add_speaker

```bash
python -m matcha.add_speaker \
  --style-encoder-ckpt logs/train_style_encoder/.../checkpoint.ckpt \
  --matcha-ckpt logs/train/v6/checkpoint_epoch=519.ckpt \
  --csv data/extra-speakers-24k/train.csv \
  --output logs/train/v6/checkpoint_with_new_speaker.ckpt
```

The script prints the new speaker's ID at the end.

### 3. Synthesise

```bash
python -m matcha.cli \
  --checkpoint_path logs/train/v6/checkpoint_with_new_speaker.ckpt \
  --vocoder vocos \
  --spk <new_speaker_id> \
  --text "Hello, this is the new speaker."
```

---

## Inference internals

At inference time, the lookup table calls are replaced with ASE and RSE:

```python
pred_enc_emb = acoustic_style_encoder(mel, mel_mask)
pred_dur_emb = rhythm_style_encoder(pred_enc_emb)
```

Both predicted embeddings are passed to `matcha.encoder(...)` in place of the lookup table outputs.

---

## Why the RSE uses only the ASE output

The RSE's job is to predict a 64-number duration embedding from a mel recording. The ASE already extracts a rich speaker identity vector from the same mel. Since the duration embedding captures speaking rate and rhythm style — both properties of the speaker's identity — projecting the ASE output linearly into the duration embedding space is sufficient.

Earlier designs fed phoneme content vectors (`x_dp`) and MAS durations (`logw_mas`) into the RSE alongside the ASE output. Both turned out to be problematic:
- `x_dp` is speaker-neutral (same sentence = same vectors regardless of speaker), so it added no useful signal
- `logw_mas` at inference time had to be computed by running Matcha's encoder with a zeroed duration embedding, making it garbage input

Dropping both simplifies the architecture and removes a broken inference path.
