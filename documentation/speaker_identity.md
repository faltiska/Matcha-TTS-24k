# Voice characteristics that make speakers unique:

1. **Fundamental Frequency (F0)** — the rate of vocal fold vibration; perceived as pitch
2. **Formants** — resonant frequencies of the vocal tract that shape timbre and vowel quality
3. **Intonation** — the rise and fall of pitch across an utterance
4. **Stress & emphasis** — prominence placed on specific syllables or words
5. **Speaking rate & rhythm** — the speed and temporal patterning of speech
6. **Accent** — phonetic patterns tied to regional or linguistic background

3 A model that captures speaker identity well

## Input
- **Mel spectrogram** (as given) — captures formants and spectral texture well
- **+ explicit F0 contour** as an additional input channel — F0 is poorly represented in mel spectrograms, and it's critical for capturing intonation and stress

## Backbone Architecture: ECAPA-TDNN
Currently the strongest architecture for speaker embeddings. It uses:
- **Multi-scale TDNN blocks** — captures local (formants, accent) and longer-range (rhythm, rate) patterns simultaneously
- **Squeeze-Excitation** — helps the model weight which frequency bands matter most per speaker
- **Residual connections** — for training stability

A **Transformer-based encoder** (e.g. WavLM fine-tuned) is a strong alternative, especially for capturing intonation and rhythm which are long-range dependencies.

## Pooling: Attentive Statistics Pooling
Aggregates the full temporal sequence into a fixed-size embedding by learning *which frames matter most* — naturally picks up on stress and emphasis patterns.

## Loss Function
- **AAM-Softmax** (Additive Angular Margin) or **GE2E loss** — both push embeddings from the same speaker together and different speakers apart in the embedding space

## One Caveat
Some of your 6 characteristics (rate, stress, intonation) **vary utterance-to-utterance** for the same speaker. The model will capture a speaker's *tendencies* in these dimensions rather than any single instance — which is actually what you want for a multi-speaker model.

## Summary

| Component | Captures |
|---|---|
| F0 input channel | F0, intonation |
| Multi-scale TDNN / Transformer | Formants, accent, rhythm |
| Attentive pooling | Stress, emphasis |
| AAM-Softmax | Speaker separability |

Would you want to go deeper on any piece — like the training pipeline or how to condition a multi-speaker TTS decoder on these embeddings?