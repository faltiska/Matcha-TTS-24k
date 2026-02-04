# Matcha-TTS Components

## What Each Module Does During Inference

## Text Processing
**Text Encoder** - Takes your written text and converts it into phonetic features that capture what sounds to make and their 
linguistic context. Contains a transformer-based encoder that processes phoneme sequences.

**Duration Predictor** - A simple CNN that predicts how long each phoneme should last. Takes the encoder's contextual features 
and outputs timing information.

## Audio Generation Pipeline

**Alignment** - Uses the duration predictions to create a timing map that aligns phonetic features with mel-spectrogram time 
frames. This tells the system exactly when each sound should occur.

**Flow Matching Decoder** - A diffusion-style model that generates the final mel-spectrogram. It starts with random noise and 
uses the aligned phonetic features as conditioning information to gradually transform the noise into realistic speech patterns 
through ODE integration (a mathematical process that solves the transformation step-by-step). The conditioning tells it what 
sounds to generate and when.

**Speaker Embedding** - For multi-speaker models, speaker characteristics are concatenated to the feature channels at multiple 
points in both the encoder and decoder, allowing the same text to be spoken in different voices.

## Audio Output
**Vocoder** - Converts the mel-spectrogram into actual audio waveform:
- **HiFiGAN** - Older vocoder for 22kHz audio
- **Vocos** - Newer vocoder for 24kHz audio (better quality)

## Key Controls
- **Number of steps** - More steps = higher quality but slower generation
- **Temperature** - Controls randomness/variation in the output
- **Speaking rate** - Speed up or slow down speech by scaling durations
- **ODE solver method** - Different mathematical approaches for the transformation process

## The Complete Flow
1. Text → phonetic features (encoder) + timing predictions (duration predictor)
2. Alignment maps phonetic features to mel-spectrogram timeframes
3. Decoder uses aligned features to condition an ODE integration process: noise → realistic mel-spectrogram
4. Vocoder converts mel-spectrogram → playable audio

The key insight: The encoder provides the "what and when" (phonetic content + timing), while the decoder provides the "how" 
(detailed acoustic realization through conditioned generation).

## Where to Find Each Component

- **Text Encoder & Duration Predictor**: `matcha/models/components/text_encoder.py`
- **Flow Matching Decoder**: `matcha/models/components/flow_matching.py` and `matcha/models/components/decoder.py`
- **Alignment**: `matcha/models/matcha_tts.py` and `matcha/utils/monotonic_align/`
- **Main Model**: `matcha/models/matcha_tts.py`
- **HiFiGAN Vocoder**: `matcha/hifigan/models.py`
- **Vocos Vocoder**: `matcha/vocos24k/wrapper.py`
- **CLI Interface**: `matcha/cli.py`