# Style Encoder

## Purpose

The Style Encoder lets you add a new voice to a trained Matcha-TTS model from a small set of recordings
(~50 Harvard Sentences), without retraining the full model.

Matcha-TTS v23 represents each speaker as two learned 128-number vectors:
- one drives the text encoder, shaping how phonemes are mapped to mel frames
- one drives the duration predictor

For a known speaker those vectors come from two lookup tables trained over 4+ days. The Style Encoder learns
to predict them from a short recording, so a new speaker does not have to go through full training.

---

## Architecture

Two independent networks, one per vector. They share no weights.

```
mel spectrogram (100 frequency bins x time frames, 5.3ms per frame)
  |
  |-- acoustic network                     |-- rhythm network
  |     4 x (Conv1d kernel 5 -> SiLU)      |     4 x (Conv1d kernel 5 -> SiLU)
  |     mean + std over time               |     mean + std over time
  |     linear projection                  |     linear projection
  |                                        |
  +--> encoder embedding (128)             +--> duration embedding (128)
```

Each network reads the whole mel, summarizes it over time by both its average and its variability, then
projects that summary into the vector the main model expects. Padded frames are masked out before every
convolution and excluded from the pooling.

Both networks use the same width and depth, taken from one pair of configuration values. Together they hold
about 2.4 million numbers, of which the conv stacks are nearly all.

### Why two networks instead of one with two heads

The main model keeps its encoder and its duration predictor strictly separated: neither one's loss can change
the other's weights. The Style Encoder mirrors that.

A single shared conv stack with two projection heads would be about half the size, but both losses would
train that stack, so each would shape the features the other depends on. Before this was split, both losses
did measurably reach the shared stack. Two networks also give the two vectors independent capacity, which is
why StyleTTS2 uses two separate networks as well.

---

## Training

The Matcha checkpoint is frozen and held in evaluation mode. None of its weights change, and its dropout
stays off. Only the Style Encoder is trained. The frozen model's decoder is deleted after loading, since
distillation only needs the text encoder and the two speaker embedding tables.

### What happens for each batch

```
1. Both networks read the mel                        -> predicted encoder vector, predicted duration vector

2. Frozen encoder, real encoder vector + real duration vector
                                                     -> reference mel output, reference log-durations

3. Frozen encoder, PREDICTED encoder vector          -> predicted mel output

4. Frozen encoder, REAL encoder vector
                   + PREDICTED duration vector       -> predicted log-durations

5. Acoustic loss: smooth L1 between the two mel outputs, each weighted per mel bin
   Rhythm loss:   smooth L1 between the two sets of log-durations
   Total loss:    the sum of the two
```

The two losses are added into one number for the optimizer, exactly as the main model adds its three. They
do not interact. Each trains only its own network: the acoustic loss cannot reach the rhythm network and the
rhythm loss cannot reach the acoustic one, because the networks share no weights and because step 4 keeps
acoustic error out of the rhythm path.

Each loss is averaged over its own valid output elements, so the 100 mel values per phoneme do not outvote
the single duration value by sheer count. There are no loss weights. With the two networks separated,
neither loss competes with the other for shared capacity, so there is nothing to rebalance.

Biases are excluded from weight decay, matching the main model. Decaying the projection biases would pull
every predicted vector towards the origin of the embedding space, away from the non-zero centre of the
trained tables.

Validation runs every five epochs. Checkpoints are selected by the lowest `val/style_loss`, not by epoch.

### Why step 4 uses the real encoder vector

Step 4 looks redundant next to step 3, and it costs an extra pass through the frozen encoder. The reason is
that the duration predictor does not read the raw phoneme stream. It reads the encoder's own output, with
the gradient path cut.

If the durations came from step 3, the duration predictor's input would already carry whatever error the
predicted encoder vector has. The cut gradient path then stops the rhythm loss from reaching the network
responsible for that error, so the rhythm network becomes the only place the gradient can land. It would
learn a vector biased to cancel an acoustic error rather than one that describes the speaker's rhythm. That
compensation goes stale as soon as the acoustic network improves, and it does not survive the independent
averaging that adding a new speaker performs across recordings.

Measured on the v23 checkpoint, at the acoustic accuracy the previous training run reached, this shifted
durations by about as much as the genuine rhythm error itself. Roughly half of what the rhythm loss measured
was not rhythm error.

### Why the acoustic loss weights each mel bin

The main model's prior loss multiplies every mel bin by a fixed perceptual weight before measuring error,
biasing it towards the frequencies human hearing is most sensitive to. The frozen encoder being distilled was
trained under that weighting, so the acoustic loss uses the same weighting.

Without it the Style Encoder would spread its capacity evenly across all 100 bins and trade error between
them differently than the model it has to reproduce.

### Reading the charts

`error_quantiles/acoustic_*` and `error_quantiles/rhythm_*` are the per-element absolute errors that drive the
two losses. Percentiles run from p25 to p100, matching the main model's diagnostics. The acoustic ones are
measured on the perceptually weighted values, the same ones the loss sees, so `acoustic_p100` can be read
directly as a threshold value.

`error_quantiles/acoustic_emb_dist_*` and `error_quantiles/rhythm_emb_dist_*` show how far the predicted
vectors land from the stored ones. They are not optimized. Nothing pulls a predicted vector towards a stored
one, by design: the Style Encoder learns only from the effect a vector has on the main model.

Those two distance charts do **not** report a plain distance, because a plain distance misleads. A speaker
vector reaches the main model through exactly one layer, which turns its 128 numbers into the scale and shift
values applied inside the encoder or the duration predictor. That layer reacts strongly to a handful of
directions and barely at all to the rest: in the v23 checkpoint, 113 of the duration predictor's 128 input
directions carry under 5 percent of the effect of the strongest one, and 101 of 128 do so in the text encoder.

So a predicted vector can sit far from the stored one and still produce identical output, because the
difference lies where the model does not look. Measured on v23, an error of plain size 0.2311 changes
durations by under 0.0015 when it lies in an ignored direction and by 0.4069 when it lies in an influential
one. A plain distance reports 0.2311 for both.

The charts therefore push the error through that same layer first and report the resulting change in the
scale and shift values, which weights every direction by how much it actually matters. Startup prints the
yardstick for each chart, so the numbers can be read without guesswork:

```
[🍵] Two different speakers differ by 0.5435 on the acoustic_emb_dist chart.
[🍵] Two different speakers differ by 0.6791 on the rhythm_emb_dist chart.
```

An error approaching that value means the prediction is as wrong as naming a different speaker. An error far
below it means the main model barely notices.

This is also why a loss comparing vectors directly would be a mistake. It would treat all 128 directions as
equally important and spend capacity on the 113 that change nothing, competing with the 15 that do.
Comparing effects does that weighting for free.

One consequence to keep in mind: as the main model trains, magnitude migrates out of the speaker embedding
tables and into these projection layers, so the same vector error causes a larger effect over time. Measured
across 235 epochs of v23, the duration side became 90 percent more sensitive and the encoder side 29 percent.
A later main-model checkpoint is a harder target, and the two yardstick values above grow with it.

---

## Files

| File | Description |
|------|-------------|
| `matcha/models/style_encoder.py` | `SpeakerEmbeddingPredictor`, `StyleEncoder`, `StyleEncoderLightningModule` |
| `matcha/train_style_encoder.py` | Training entry point (mirrors `train.py`) |
| `matcha/add_speaker.py` | Writes a new speaker into a copy of a Matcha checkpoint |
| `configs/model/style_encoder/default.yaml` | Model hyperparameters and the Matcha checkpoint to distill |
| `configs/train_style_encoder.yaml` | Top-level Hydra config, validation, and checkpoint selection |
| `tests/test_style_encoder.py` | Unit tests |

---

## Configuration

`configs/model/style_encoder/default.yaml` holds:
- `matcha_checkpoint_path` — the trained Matcha checkpoint to distill from
- `ase_hidden_channels`, `ase_n_layers` — width and depth, used for both networks
- `acoustic_loss_threshold`, `rhythm_loss_threshold` — the two Huber transition points

The mel feature count and the speaker vector size are read from the Matcha checkpoint at startup, so the
Style Encoder cannot silently use dimensions that differ from the model it conditions.

### The two thresholds

Each threshold is the error size above which its loss switches from quadratic to linear, so that large
errors stop dominating the gradient. They follow the same rule as the main model's prior and duration
thresholds: set each one to the largest absolute error still present at the end of training, which is the
`error_quantiles/*_p100` chart. Placed there, the linear region only engages early in training while errors
are still large, and the loss is purely quadratic once the model has converged.

The values cannot be copied from the main model, for two reasons. The main model measures its predicted mel
against a ground truth mel, while these losses measure only the drift caused by swapping a real vector for a
predicted one, which is far smaller — in the v23 run the prior error reached p90 = 0.12, while the Style
Encoder's converged acoustic error reached p90 = 0.0039. And the two use different loss functions:
`huber_loss(delta=d)` equals `d * smooth_l1_loss(beta=d)`, so the same number means a different loss scale in
each place.

The values currently in the config are the ones the Style Encoder was previously trained with. The earlier run
only recorded percentiles up to p90, so there is no converged maximum to tune against yet. p100 is now logged;
read it off the next run and raise both thresholds to it.

---

## Running training

```bash
source .venv/bin/activate
python -m matcha.train_style_encoder
```

Note that existing Style Encoder checkpoints cannot be resumed. The two-network split renamed every
parameter, and earlier checkpoints also produce 96-number vectors where v23 expects 128. `ckpt_path` is
`null` for that reason.

---

## Adding a new speaker

### 1. Prepare the corpus

Prepare it as you would for training:
```bash
./prepare_corpus.sh configs/data/extra-speakers-24k.yaml
```

This produces the precomputed mels that `add_speaker` reads. It needs the fine-resolution ones, saved
alongside the normal ones as `<name>.fine.npy`.

### 2. Run add_speaker

```bash
source .venv/bin/activate
python -m matcha.add_speaker \
  --style-encoder-ckpt <trained-style-encoder.ckpt> \
  --matcha-ckpt logs/train/v23/checkpoint_epoch=234.ckpt \
  --csv data/extra-speakers-24k/train.csv \
  --output checkpoint_with_new_speaker.ckpt
```

Recordings are processed one at a time. For each one both networks predict a vector; the two sets of
predictions are then averaged independently across all recordings, appended as a new row to each of the two
lookup tables, and written to a copy of the Matcha checkpoint. The speaker count in the checkpoint's
hyperparameters is updated to match.

The script prints the new speaker's ID at the end.

### 3. Synthesise

```bash
source .venv/bin/activate
python -m matcha.cli \
  --checkpoint_path checkpoint_with_new_speaker.ckpt \
  --spk 15 \
  --text "Bună, numele meu este Daria."
```

Because the vectors are written into the lookup tables, synthesis needs nothing from the Style Encoder. The
new speaker is an ordinary speaker from then on, including for blending with other speakers.

---

## Known gaps

**Validation cannot measure what matters.** Both `train.csv` and `validate.csv` contain all 15 speakers, so
validation only ever shows the Style Encoder new sentences from voices it already knows. It never shows it a
new voice, which is the only thing the Style Encoder exists to handle.

Holding 2 or 3 whole speakers out of Style Encoder training would fix that. The main model still knows those
speakers, so their real vectors remain available to compare against. Until then, changes to the Style Encoder
cannot be told apart by their numbers.

A cheap memorising check to run alongside it: for a held-out voice, measure how far the prediction lands from
the nearest of the 15 known vectors. Two real speakers sit about 1.2 apart in plain per-number terms. A
prediction landing much closer than that to a known vector has snapped onto a memorised one.

**Only 15 speakers.** No loss will teach a general rule for turning any voice into a vector from 15 examples.
More speakers is the only real fix.

**Shallow conv stacks.** Both networks are plain Conv1d + SiLU: no normalization, no residuals, and a short
receptive field. That limits how much speaker character they can capture across longer spans, and makes them
hard to deepen safely. `text_encoder.py` already has `LayerNorm` and `ConvSiluNorm` to mirror.

StyleTTS2's style encoder is the deeper, normalized version of the same idea and a reasonable blueprint:
- it treats the mel as a 1-channel 2D image and runs residual blocks, each halving both the frequency and
  time axes while doubling channels up to a cap — repeated time halving is how it reaches a long receptive
  field with few blocks, as an alternative to going deeper or using dilations
- every conv is wrapped in `spectral_norm`, and blocks optionally use `InstanceNorm2d`
- each block adds a learned shortcut and rescales the sum to keep unit variance
- it finishes with a mean-only global pool, so on pooling we are already ahead with mean+std

One structural choice of theirs we deliberately do not copy: their style encoder is trained end-to-end,
adversarially and with diffusion, and so co-defines the style space. Ours distills a frozen model, so our
style space is capped by what the trained lookup tables already encode. That is a deliberate trade for a
simpler, decoupled setup that cannot destabilize the main model.
