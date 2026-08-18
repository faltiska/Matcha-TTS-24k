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

### Downsampler mechanism — settled, the box downsampler stays

For experiment v22 I took the fully trained Encoder and Duration Predictor from v21 and an empty Decoder, then trained for 2000 epochs with a `triangular_downsample` method, a centered [1, 2, 1] / 4 filter, in place of the [1, 1, 1] / 3 box filter. That was the only code change.
The speech did come out sharper, confirming that the box filter has a smoothing effect. But it came out worse: audible artifacts and harshness that v21 did not have, and an MCD score 0.1dB worse after 2000 epochs than v21 reached in 1300.
`triangular_downsample` has been deleted. `box_downsample` in @file:model.py is the only downsampler now, and both training and inference use it.

Two separate mistakes led to that experiment, and both are worth remembering.

The criterion I selected the filter by was meaningless. I picked the triangular filter because it had lower MAE and MSE against the corpus standard mels. But both mel resolutions are extracted with the same 1024 sample analysis window and differ only in hop, so the standard frames land on the same sample positions as the even numbered fine frames: plain decimation, `[:, :, ::2]`, reproduces the standard mel exactly, at zero error. Ranking filters by that error only ranks them by how little they filter, and the winner of that contest is doing nothing at all. See the Mel Analysis Window section in components.md.

Average error is also the wrong quantity to minimize for this Decoder. It doesn't tell you how the error is distributed, and for the Decoder that's what matters.
The box downsampler is consistently wrong in the same direction: heavier averaging always pulls values toward their neighbours, so its output is systematically a bit flatter than the true standard mel, by a roughly similar amount everywhere. The triangular filter was closer on average but its remaining error depended on what the signal was doing locally — near-zero where the mel is flat, larger where it's moving fast. Same or better average, less predictable frame to frame.
Since use_mu_prior is true, the Decoder isn't a general-purpose mapper — it learns a correction from the assembled mel toward the ground truth. A correction learner handles a consistent bias almost for free: it learns "always push a little this way." It handles a varying error much worse, because the right correction now differs from frame to frame and it has to infer which case it's in from context. So swapping a large predictable error for a smaller unpredictable one made the Decoder's job harder even though its input was objectively more accurate.
That also explains the shape of what I was hearing: the harshness wasn't uniform, it was concentrated where the mel changes quickly, because that's where the triangular filter's error was largest and least predictable.
The mechanism is still only a hypothesis, and it is testable: measure the error of both downsamplers against the standard mel separately on frames where the mel is changing fast versus slow. If triangular wins on average but loses on the fast-changing frames, that's the mechanism. I did not run that test, I reverted instead.

Here are some improvement ideas based on that observation. Any future filter change needs a criterion that is not "error against the corpus standard mel", since decimation already scores zero there.
1. Average in energy, not in log. The corpus standard mel is an energy measurement made by the STFT. Your downsamplers average log-compressed values, which always understates the result, and understates it most where the mel is changing fast — exactly the frames where I think the error is hurting you. Denormalize, exponentiate to mel energy, average there, then go back to log and renormalize. This attacks the uneven-error problem at its source rather than trading one filter shape for another. Cost is an exp and a log per frame, negligible on GPU, and no architecture change.
2. Fit the filter instead of designing it. The best 3-tap or 5-tap filter is a least-squares problem with a closed-form answer — no training. Two important details: fit it per mel bin rather than one filter for all bins, and fit it from the model's assembled mel toward the ground truth standard mel, never from corpus fine mels toward corpus standard mels, because that second problem has the trivial answer of decimation and tells you nothing. Fitted against the error the Decoder actually receives, this gives you the optimum in that family and ends the guessing between hand-picked kernels.
3. Tell the Decoder which case it's in. If the conditioning error varies with how fast the mel is moving, the Decoder currently has to infer that from context. Hand it the local rate of change as an extra conditioning channel. One channel, near-free, and it converts an unpredictable error into a predictable one, which is what the correction learner is good at.
4. Weight the Decoder loss perceptually. compute_loss uses plain MSE on the velocity, treating all mel bins as equally important, while your prior loss already uses perceptual_mel_weights. Harshness lives in specific bands, and right now the loss has no reason to prioritize them. Reusing the same weighting is a small change.

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
