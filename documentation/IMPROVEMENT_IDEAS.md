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

### Downsampler mechanism

For experiment v22, I took the fully trained Encoder and Duration predictor from v21, and an empty Decoder. Then I trained for 2000 epochs with a new Downsample method, see @file:model.py  triangular_downsample.
I tested the traingular downsampler on fine resoution mels from the corpus. IT produces mels spectrograms much closer to the standard resolution mels from the corpus. And one subjective tests, the TTS produced speech sounds sharper that in v21, confirming that the old downsampler has a smoothing effect.
I would have expcted the speech to sound better and measure better in my MCD tests. It does not. I can hear artifacts and harshness that was not there in v21. And the MCD score after 2000 epochs is 0.1dB worse that it was on v21 where I trained for just 1300 epochs.
There id no other code change. Only tyhe triangular downsampler replaced the previous downsampler.
Still, the quality is worse, both subjectively and measured.

The tests showed triangular_downsample has lower MAE and MSE against the corpus standard mels. That means lower error on average. But it doesn't tell you how the error is distributed, and for the Decoder that's what matters.
The old downsample was consistently wrong in the same direction: heavier averaging always pulls values toward their neighbours, so its output was systematically a bit flatter than the true standard mel, by a roughly similar amount everywhere. triangular_downsample is closer on average but its remaining error depends on what the signal is doing locally — near-zero where the mel is flat, larger where it's moving fast. Same or better average, less predictable frame to frame.
Since use_mu_prior is true, the Decoder isn't a general-purpose mapper — it learns a correction from the assembled mel toward the ground truth. A correction learner handles a consistent bias almost for free: it learns "always push a little this way." It handles a varying error much worse, because the right correction now differs from frame to frame and it has to infer which case it's in from context. So swapping a large predictable error for a smaller unpredictable one can make the Decoder's job harder even though the input is objectively more accurate.
That also explains the shape of what you're hearing: harshness isn't uniform, it's concentrated where the mel is changing quickly, because that's where the new filter's error is largest and least predictable.
If that's right, it's testable: measure the error of both downsamplers against the standard mel separately on frames where the mel is changing fast versus slow. If triangular wins on average but loses on the fast-changing frames, that's the mechanism.

Here are some improvement ideas based on that observation.
1. Average in energy, not in log. The corpus standard mel is an energy measurement made by the STFT. Your downsamplers average log-compressed values, which always understates the result, and understates it most where the mel is changing fast — exactly the frames where I think the error is hurting you. Denormalize, exponentiate to mel energy, average there, then go back to log and renormalize. This attacks the uneven-error problem at its source rather than trading one filter shape for another. Cost is an exp and a log per frame, negligible on GPU, and no architecture change.
2. Fit the filter instead of designing it. You have both resolutions in the corpus, so the best 3-tap or 5-tap filter is a least-squares problem with a closed-form answer — no training. Two important details: fit it per mel bin rather than one filter for all bins, and fit it starting from the model's assembled mel rather than from corpus fine mels, so you're optimizing the error the Decoder actually receives. This gives you the optimum in that family and ends the guessing between two hand-picked kernels.
3. Tell the Decoder which case it's in. If the conditioning error varies with how fast the mel is moving, the Decoder currently has to infer that from context. Hand it the local rate of change as an extra conditioning channel. One channel, near-free, and it converts an unpredictable error into a predictable one, which is what the correction learner is good at.
4. Weight the Decoder loss perceptually. compute_loss uses plain MSE on the velocity, treating all mel bins as equally important, while your prior loss already uses perceptual_mel_weights. Harshness lives in specific bands, and right now the loss has no reason to prioritize them. Reusing the same weighting is a small change.