# Don't log losses per step anymore, I never look at those.

# Use FiLM within the Encoder blocks instead of concatenation
As done in ConvDurationPredictor3

# CFM / Diffusion improvement 

## 1. The "Beta" Schedule (Shifted Sampling)
Instead of weighting the loss (which keeps the gradients the same but scales them), try changing **which** timesteps the model sees more often.
In mel-spectrogram generation, the model often struggles with the fine details that emerge near.
Instead of `torch.rand`, sample  from a **Beta distribution**.
This forces the model to spend more "brain power" on the complex parts of the trajectory.

```
# Instead of t = torch.rand(...)
# alpha > 1 shifts sampling towards t=1 (the target)
# alpha=1.5 or 2.0 is a good starting point
dist = torch.distributions.Beta(1.5, 1.0) 
t = dist.sample([b, 1, 1]).to(mu.device)
```

## 2. Auxiliary Reconstruction Loss

Right now, your model only learns the "velocity" ().
It doesn't actually "see" the final mel-spectrogram it's trying to create until inference.
You can add a secondary loss that forces the predicted velocity to be consistent with the target.
It is a secondary objective that supervises the model's ability to recover the clean target ($x_1$) 
from any noisy intermediate state ($y$).
While the primary Flow Matching loss penalizes the error in the predicted velocity (the "vector" or "direction"), 
the reconstruction loss penalizes the error in the implied final endpoint. 
It effectively transforms the network's velocity prediction back into a data-space estimate and compares it directly 
against the ground truth mel-spectrogram.

Technical Benefits:
Global Structural Consistency: Velocity loss is local and differential. 
Reconstruction loss provides a global signal, ensuring the predicted trajectory actually lands on a valid 
mel-spectrogram manifold.
Gradient Stabilization: It acts as a regularizer. 
By forcing the model to "see" the destination at every timestep $t$, it reduces the variance of the gradients, 
especially in the high-noise regimes (low $t$) where the optimal velocity is harder to estimate.
Spectral Feature Alignment: In audio tasks, Flow Matching can sometimes produce high-frequency artifacts. 
A reconstruction loss (especially using L1) encourages the model to preserve the harmonic structure and energy 
distribution of the original speech.

Implementation Logic:
Algebraic Projection: Use the current noisy sample and the network's output to solve for the implied clean target.
Masking: Apply the same sequence mask used in the flow loss to ignore padding.
Loss Calculation: Compute the Mean Absolute Error (L1) between the implied target and the actual ground truth.
Weighted Sum: Add this value to your primary loss, scaled by a hyperparameter (e.g., $0.5$).

```
# Inside compute_loss
pred = self.estimator(y, mask, mu, t.squeeze(), spks)
# Calculate the implied target
x1_pred = y + (1 - t) * pred
# Combined loss: Flow loss + Reconstruction loss
loss_flow = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
loss_recon = F.l1_loss(x1_pred * mask, x1 * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])

loss = loss_flow + 0.5 * loss_recon # Lambda of 0.5 is usually safe
```

## 3. Switch to L1 Loss for the Flow

You are currently using `F.mse_loss`. 
While MSE is mathematically "clean" for Gaussian processes, mel-spectrograms are notoriously full of outliers and sharp peaks.
**L1 loss** (Mean Absolute Error) is significantly more robust for audio tasks.
It prevents the model from over-correcting for rare, high-energy spectral peaks at the expense of the subtle textures.


# Dropout on mu_y to reduce harsh speaker characteristics

The CFM decoder receives `mu_y` (the encoder's output) as its conditioning signal.
`mu_y` carries the spectral character of the training recordings, including harsh or metallic voice qualities,
because the encoder learned to reproduce those characteristics to minimize prior loss.

Adding dropout to `mu_y` during training forces the CFM to generate good speech even when the conditioning
signal is slightly corrupted. Over time, the model learns to rely more on its own learned priors about
what speech should sound like, and less on faithfully reproducing every spectral detail of `mu_y`.
At inference time dropout is off, but the model will have learned to be less dependent on the exact
spectral details — producing a cleaner, more "ideal" version of the speaker's voice.

Implementation: in `matcha_tts.py`, apply `F.dropout(mu_y, p=0.1, training=self.training)` before
passing `detached_mu_y` to `self.decoder.compute_loss()`.

This requires retraining from scratch or fine-tuning from a checkpoint.
Start with p=0.05 and measure MCD — lower MCD is not the goal here, better perceptual quality is.


# Validation tools like WER, MCD, UTMOS

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


## Code changes I can consider in the future (not now!)
Use a LR scheduler
