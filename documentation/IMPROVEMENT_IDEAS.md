# CFM / Diffusion improvement 

## 1. The "Beta" Schedule (Shifted Sampling)
Instead of weighting the loss (which keeps the gradients the same but scales them), try changing **which** timesteps the model sees more often.
In mel-spectrogram generation, the model often struggles with the fine details that emerge near destination.
Instead of `torch.rand`, sample  from a **Beta distribution**.
This forces the model to spend more "brain power" on the complex parts of the trajectory.

```
# Instead of t = torch.rand(...)
# alpha > 1 shifts sampling towards t=1 (the target)
# alpha=1.5 or 2.0 is a good starting point
dist = torch.distributions.Beta(1.5, 1.0) 
t = dist.sample([b, 1, 1]).to(mu.device)
```

# Validation tools like WER, UTMOS

Evaluating a TTS model like Matcha-TTS usually falls into three buckets: 
**Intelligibility** (does it say the right words?) 
**Fidelity** (how close is it to the original file?) 
**Naturalness** (does it sound like a human?)

I have already fidelity using MCD, and I am using it to compare progress over multiple checkpoints.
Here is how you can calculate the other metrics using Python.

---

### 1. Intelligibility: Word Error Rate (WER)

WER measures how well an Automatic Speech Recognition (ASR) model can "understand" your synthesized speech. 
If the ASR model can't transcribe it correctly, a human probably won't either.

* **Tools:** `openai-whisper` (for transcription) and `jiwer` (for the calculation).

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


## Code changes I can consider in the future (not now!)
Use a LR scheduler

# Improved Losses

## Apply perceptual weights after calculating the elemental loss

The perceptual mel-bin weights are currently applied to the prediction and target before calculating the Huber loss. This weights the residual rather than the resulting loss.

Inside Huber's quadratic region, multiplying a residual by a weight causes its loss contribution to be multiplied by the square of that weight. It also changes the residual value at which Huber switches from quadratic to linear independently for each mel bin.

Calculate Huber with no reduction first, then multiply its elemental output by the perceptual mel-bin weights before reducing it. This makes the configured weights the actual loss multipliers and preserves one common Huber transition point across all frequencies.

This must be tested against the current behavior rather than assumed to be better, because the model may benefit from the stronger weighting it currently receives.

## Perceptually weight the Decoder loss

The Prior loss prioritizes perceptually important mel-frequency bins, while the Decoder's Conditional Flow Matching (CFM) loss treats every mel bin equally.

Apply the same gentle perceptual curve to the Decoder's elemental squared errors before reducing them. The Decoder has an independent gradient path, so this cannot create a conflict with the Prior loss.

This idea is also listed under the downsampler improvements, but it applies independently of which downsampler is used.

## Weight Decoder transition frames more strongly

The Decoder's main job is to smooth the assembled mel spectrogram, especially around coarticulation and abrupt transitions. Its current loss gives stationary frames and rapidly changing frames equal importance.

Measure the local change in the target mel spectrogram and give changing frames a moderately larger loss weight. Smooth, clip, and normalize the weights so noise or silence boundaries cannot dominate training. A narrow range such as 0.75 to 1.25 would be a conservative starting point.

This retains one Conditional Flow Matching objective instead of adding a potentially competing temporal loss.

## Introduce duration-balanced Prior loss gradually

Switching directly from frame-balanced to symbol-balanced Prior loss produced a large loss spike even when enabled after epoch 214.

Generalize the duration divisor with an exponent. An exponent of zero reproduces the original frame-balanced loss, while an exponent of one gives every symbol equal total weight. Gradually increase the exponent after Monotonic Alignment Search (MAS) becomes reliable.

This should preserve the eventual benefit of equal symbol weighting while avoiding an abrupt change to the Encoder's gradients and alignments.

## Weight alignment-derived losses by alignment confidence

The Prior and Duration Predictor losses both depend on hard alignments produced by Monotonic Alignment Search. An uncertain or incorrect alignment can train the Encoder toward the mistake, which then changes the next alignment and creates a positive feedback loop.

Estimate how decisively the selected symbol matches each frame compared with nearby alternatives. Detach this confidence from gradient calculation and use it to reduce, but never eliminate, the contribution of uncertain assignments.

Use clipped weights with a substantial minimum and normalize them to keep the average gradient scale stable. Confidence weighting could also be combined with the gradual duration-balancing schedule.

## Balance losses across symbol types and speakers

Equal weighting per symbol occurrence does not give equal influence to symbol types. Frequent phonemes, coarticulation symbols, or speakers with more recordings can still dominate training.

Measure errors by symbol type, symbol role, and speaker first. If significant imbalance remains, test clipped inverse-square-root frequency weights normalized to an average of one. Avoid unrestricted inverse-frequency weighting because very rare examples may have noisy targets.

## Stratify Conditional Flow Matching timesteps

Conditional Flow Matching currently samples each training timestep independently and uniformly. Random batches can consequently overrepresent some parts of the flow trajectory and omit others.

Distribute the samples in each batch across the complete trajectory while preserving a uniform distribution overall. This does not change the objective, but should reduce gradient variance.

Before changing the distribution itself, log Decoder errors by timestep. If particular trajectory regions consistently remain harder, compare stratified uniform sampling with the existing idea of intentionally biased timestep sampling.
