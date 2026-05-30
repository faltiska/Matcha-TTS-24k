# Style Encoder

## Purpose

The Style Encoder lets you add a new voice to a trained Matcha-TTS model using a small set of recordings 
(~50 Harvard Sentences), without retraining the full model.

Matcha-TTS represents each speaker as two learned vectors:
- One drives the text encoder, shaping how phonemes are mapped to mel frames
- One drives the duration predictor

For a known speaker, these vectors come from two lookup tables trained over 4+ days. The Style Encoder learns to 
predict them directly from a short audio recording, so a new speaker does not need to go through full training.

---

## Architecture

### StyleEncoder

Predicts both speaker embeddings from a mel spectrogram.

```
mel spectrogram  (frequency bins × time frames)
  → stack of convolutional layers + ReLU, masked
  → mean + standard deviation over time (ignoring padding)
  → two linear projections
        • encoder embedding   (for the text encoder)
        • duration embedding  (for the duration predictor)
```

The conv stack extracts voice characteristics, summarizes them over time with both their average and their variability, 
then two linear heads project that summary into the two embeddings that Matcha's encoder and duration predictor expect.

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

The two losses teach the StyleEncoder to predict embeddings that make the encoder reproduce both the phoneme-to-mel 
mapping (acoustic) and the predicted durations (rhythm) it would have produced with the real embeddings.

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

`configs/model/style_encoder/default.yaml` holds:
- the path to the trained Matcha checkpoint to distill from
- the mel feature count and speaker embedding dimension — both must match that checkpoint
- the conv stack's width and depth

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

Weakness: shallow conv stack, no normalization

The current stack is plain Conv1d+ReLU, with a short receptive field, no normalization, no residuals. Two issues: receptive field too short to capture speaker character across longer spans, and no normalization makes a deeper stack harder to train. Fixes, in order of effort:
• Add normalization + residuals to the existing conv blocks so you can deepen safely (your text_encoder.py already has LayerNorm and ConvSiluNorm — same idea, can mirror that style).
• Widen the receptive field, either by going deeper or by dilations.
The honest caveat I gave before still holds: these two fix timbre fidelity, which is already your good axis. They won't fix the accent — that's the entanglement/main-model story. So I'd treat them as a second, lower-priority workstream, not the headline.

#### Reference: how StyleTTS2 builds its style encoder

StyleTTS2's `StyleEncoder` is essentially the deeper, normalized version of this same idea, so it is a good blueprint for the fixes above:
- It treats the mel as a **1-channel 2D image** and runs a stack of **residual blocks** (`ResBlk`), each halving **both** the frequency and time axes (`downsample='half'`) while doubling channels (up to a cap). Halving the time axis repeatedly is how it reaches a long effective receptive field with only a few blocks — an alternative to going deeper or using dilations.
- Every conv is wrapped in **`spectral_norm`**, and the blocks optionally use **`InstanceNorm2d`** (the normalization our current stack lacks). Activation is `LeakyReLU`.
- Each `ResBlk` adds a learned `_shortcut` and rescales the sum to keep unit variance — a cheap residual recipe worth copying if we deepen.
- It finishes with a conv and **`AdaptiveAvgPool2d`** — i.e. a **mean-only** global pool. Our mean+std stats pooling is already more expressive than this, so on pooling we are ahead; the gap is purely the backbone.

Two structural choices of theirs we deliberately do **not** copy:
- StyleTTS2 uses **two completely separate encoder networks** (one acoustic, one prosodic). We use one shared conv backbone with two linear heads, which is cheaper but gives the two embedding spaces less independent capacity. If the duration/rhythm side stays weak after the backbone upgrade, splitting into two backbones is the next lever.
- Their style encoder is trained **end-to-end** (adversarial + diffusion) and *co-defines* the style space. Ours is a **distiller** that regresses the frozen lookup-table embeddings, so our style space is capped by what those vectors already encode — a deliberate trade for a simpler, decoupled training setup that cannot destabilize the main model.

The `InstanceNorm2d` + residual + variance-rescaling recipe maps cleanly onto our `text_encoder.py` style (`LayerNorm`, `ConvSiluNorm`), so we can mirror either source when we pick this up.

That leaves us with potentially three coordinated changes:
1. ~~Main model: concat → FiLM/AdaLN in the Encoder (fixes the accent; the headline).~~ Done in v20.
2. ~~Style Encoder pooling: mean → mean+std.~~ Done.
3. Style Encoder backbone: add norm/residuals, widen receptive field.
