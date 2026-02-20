# Timestep Weighting for Flow Matching Loss

## Motivation

Helps the model focus on getting the final refinement steps right, where perceptual quality matters most.

Think about the generation process from t=0 to t=1:
* **t=0**: Pure noise - the "canvas" can be rough
* **t=0.5**: Getting structure - phonemes emerging, rough spectral shape
* **t=1**: Final output - every detail matters for perceptual quality

The final 10% of the journey (t=0.9→1.0) has disproportionate impact on what the listener hears, 
but with uniform sampling, you spend equal training effort on all timesteps.

## Weighting Strategies

### Linear (t)
* Gentle, safe starting point
* 2x more weight at t=1 vs t=0
* Good for initial experiments
* Formula: `loss = t * ||v_pred - v_target||²`

### Quadratic (t²)
* Stronger emphasis on final steps
* 10x more weight at t=1 vs t=0.3
* Strong enough to matter, not so strong it destabilizes training
* Used successfully in recent diffusion TTS papers
* Formula: `loss = t² * ||v_pred - v_target||²`

## Implementation Strategy

1. Start with **linear weighting (t)** first
2. If training is stable and results improve, try **quadratic (t²)**
3. Monitor for training instability (loss spikes, NaN values)
4. Can be toggled on/off via config for ablation studies

## Why This Works for Matcha-TTS

* Compatible with Conditional Flow Matching (CFM) architecture
* No architectural changes needed - just multiply loss by weight
* Doesn't interfere with encoder/decoder independence
* Low risk, potentially high reward


# Sway sampling 

In a standard ODE (Ordinary Differential Equation) solver, you move from time  (noise) to  (data) in equal increments.
However, the "path" from noise to data is often more turbulent at the beginning and end. 
**Sway sampling** re-warps the time steps so the model spends more "focus" (smaller steps) on the high-noise regions and larger, faster steps where the data is clearer.

The core idea is to transform a linear time sequence using a power function. For Matcha, we typically want more density near  (the noise) to ensure the initial direction is correct.

I implemented and tested this using the MCD metric described beloe, and **it makes no difference**.


# Implement validation tools like WER, MCD, UTMOS

I have implemented MCD and I am using it to compare progress over multiple checkpoints.

Evaluating a TTS model like Matcha-TTS usually falls into three buckets: 
**Intelligibility** (does it say the right words?) 
**Fidelity** (how close is it to the original file?) 
**Naturalness** (does it sound like a human?)

Here is how you can calculate the most common metrics using Python.

---

### 1. Intelligibility: Word Error Rate (WER)

WER measures how well an Automatic Speech Recognition (ASR) model can "understand" your synthesized speech. 
If the ASR model can't transcribe it correctly, a human probably won't either.

* **Tools:** `openai-whisper` (for transcription) and `jiwer` (for the calculation).

---

### 2. Fidelity: Mel-Cepstral Distortion (MCD)

MCD is the metric that tells you **how close you are to the ground truth**. It compares the "texture" of the generated audio to the original recording.

* **Tools:** `mel-cepstral-distance` (a handy wrapper for DTW and MFCC calculations).
* **Note:** This requires the generated audio and ground truth audio to be the same length (it usually uses Dynamic Time Warping to align them).

---

### 3. Naturalness: Predicted MOS (Mean Opinion Score)

Historically, MOS required paying 20 humans to rate audio from 1–5. 
Today, we use AI models trained on those human ratings to "predict" the score. 
**UTMOS** is currently one of the most reliable models for this.

* **Tool:** `UTMOS` (via GitHub or Hugging Face).

---

### Summary Table of Metrics

| Metric          | Category           | Comparison Type    | Good Score                    |
|-----------------|--------------------|--------------------|-------------------------------|
| **WER**         | Intelligibility    | Reference Text     | < 5%                          |
| **MCD**         | Fidelity           | Ground Truth Audio | Lower is better (e.g., < 5.0) |
| **MOS (UTMOS)** | Naturalness        | Absolute (No Ref)  | > 4.0                         |
| **SEC**         | Speaker Similarity | Reference Speaker  | > 0.8 (Cosine Sim)            |

## TODO
- improve server.py to make it ready for prod
- add PSR script

## Code changes I can consider in the future (not now!)
Remove app.py
Do I have to return so much data from inference?
Use a LR scheduler
Improve DynamicBatchSampler fix for adding more batches if needed 
Use OGG or AAC compression instead of mp3
Train with bigvgan mels (or Try to convert Vocos mels to bigvgan mels)
See how bigvgan generates mels, maybe we want to do the same
Take 579 and run a traininig with just Brian speaker embeddings enabled

Ideas from BigVGAN v2:
Use a MAX_WAV_VALUE = 32767 instead of 32768 when computing mels (prevents int16 overflow)
Trim audio to multiple of hop lengths before converting to mel (figure out what is the benefit)
Verify if vocos was trained with a clip_val of 1e-7 (BigVGAN uses 1e-5)

Find a way to vary the prosody a little with each inference call, but just the prosody
not the timbre, like the TEMPERATURE feature was doing. 

Clean up the architecture so that the model and the lightning module are separated
- matcha/models/matcha_tts_model.py — pure nn.Module with synthesise(), no lightning, no hydra, no logging. This is essentially what MatchaTTSInfer already is.
- matcha/models/matcha_tts.py — thin LightningModule wrapper that owns the training loop (forward, losses, optimizers, validation step) and delegates to the model for everything else.