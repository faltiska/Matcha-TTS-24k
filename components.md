# Matcha-TTS Components

## What Each Module Does During Training

### Text Encoder
Takes phoneme IDs and produces two things independently:
- A mel spectrogram (one vector per phoneme, in mel-feature space) — this is the "encoder mel"
- Predicted log-durations for each phoneme (from the Duration Predictor sub-module)

The encoder never receives durations as input. Durations are a result of the encoder mel, not an input to it.

### Duration Predictor
A CNN sub-module inside the Text Encoder. It takes the encoder's phoneme features (detached, so its gradients don't affect the encoder) and predicts how many mel frames each phoneme should span. Trained against reference durations from MAS.

### Reference Durations (MAS)
The reference durations used to train the Duration Predictor come from MAS (Monotonic Alignment Search): compares the encoder mel against the ground truth mel frame-by-frame using a Gaussian log-likelihood, and finds the best alignment. This is computed during training, on the fly.

The reference durations only affect the Duration Predictor training. The encoder mel, prior loss, and decoder loss are completely unaffected, because the encoder mel is generated from phonemes only — durations play no role in building it.

### Prior Loss
Measures how close the encoder mel is to the ground truth mel (L1 distance, frame by frame). Trains the encoder to produce a mel that already resembles the target.

### Flow Matching Decoder
Receives only the encoder mel and the ground truth mel. Durations do not exist as far as the decoder is concerned — they are never passed to it, not even indirectly.

At a random timestep, it interpolates between noise and ground truth to get a noisy sample, then learns to predict the velocity field (direction from noise toward ground truth).

The encoder mel length equals the ground truth mel length — both come from the same audio file. Durations have no influence on this.

### Speaker Embeddings
Three separate embedding tables: one for the encoder, one for the duration predictor, one for the decoder. Each is concatenated to the respective module's input features.

---

## What Each Module Does During Inference

At inference there is no ground truth mel, so MAS cannot run. The Duration Predictor's output is used instead to decide how many frames each phoneme spans, which determines the total output length.

Inference applies a monotonic positions fix to the predicted durations: cumulative sum → round → enforce each position ≥ previous+1 → derive integer durations by differencing. This guarantees every phoneme gets at least 1 frame and no two phonemes start at the same frame.

The decoder still receives only the encoder mel — durations are used to build it (by repeating each phoneme's encoder vector for the right number of frames), but the durations themselves are not passed to the decoder.

---

## The Complete Training Flow

1. Phoneme IDs → encoder mel + predicted log-durations
2. MAS reference durations → duration loss against predicted log-durations
3. MAS alignment map → used for prior loss (encoder mel is already built independently)
4. Prior loss: encoder mel vs ground truth mel (L1)
5. Decoder loss: flow matching from noise toward ground truth, conditioned on encoder mel

## The Complete Inference Flow

1. Phoneme IDs → encoder mel + predicted log-durations
2. Predicted durations → monotonic fix → integer durations per phoneme
3. Durations → each phoneme's encoder vector repeated for its frame count → frame-aligned encoder mel
4. Decoder: ODE integration from noise → refined mel, conditioned on frame-aligned encoder mel
5. Vocoder: mel → audio waveform

---

## Where to Find Each Component

- **Text Encoder & Duration Predictor**: `matcha/models/components/text_encoder.py`
- **Flow Matching Decoder**: `matcha/models/components/flow_matching.py` and `matcha/models/components/decoder.py`
- **MAS Alignment**: `matcha/models/matcha_tts.py` (uses `super_monotonic_align`)
- **Precomputed Durations**: `matcha/utils/compute_durations.py`
- **Main Model**: `matcha/models/matcha_tts.py`
- **Vocos Vocoder**: `matcha/vocos24k/vocos_wrapper.py`
- **HiFiGAN Vocoder**: `matcha/hifigan/models.py`
- **CLI Interface**: `matcha/cli.py`
