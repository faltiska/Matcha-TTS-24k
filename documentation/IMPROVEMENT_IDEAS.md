# Try conformer blocks for Decoder:
down_block_type="transformer",
mid_block_type="transformer",
up_block_type="transformer",

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


# Better encoder:
1. I want to modify the encoder to separate the energy / pitch / prosody from duration prediction.
Right now, it works as follows:
Take a phonetic representation of the text, with separators in between each phoneme, let's call it X. 
X has annotation symbols too, like stress marekers and duration modifiers along with voiced phonemes. 
Separators are indiscriminatelly added between each 2 symbols, annotations or voiced.
X is represent it as symbol IDs, there is a symbol IDs embedding table. Then:
```
x = prenet(x)
x = torch.cat([x, speaker_embedding)
x = encoder(x, x_mask)
mu = proj_m(x) * x_mask
logw = proj_w(x.detach())
```

Prenet is a simple net with a kernel of 5, and 6 layers of conv + SiLU + norm.
Encoder is a transformer with attention, norm, FFN, norm.
Proj_m, the mel generator is on layer of Conv + SiLU + Conv.
Proj_w is 4 layers of conv + norm.
The Encoder result is used in 2 ways: proj_m generates a mel frame for each phoneme, proj_w predicts a duration for each.
It has a few drawbacks:
1. Annotations are present in the input for the mel generator and the duration predictor. They get assigned a duration and a mel frame, which is unfounded. They are there, so the model is modeling them...
2. Separators interject annotations and their annotated phonemes, and between last vowel and sentence terminators, like ? or !. The signal to raise the pitch accordingly is sometimes too diluted, model does not learn to read interrogations well enough.

The mel generator needs separators, no matter what.
There are transitional sounds between voiced phonemes, and the model needs a placeholder to assign them.
But the duration predictor does not need them.  Also, thy hurt pitch / energy as explained. I would rather compute this on a phonetic representation without separators.
But both proj_m and proj_w work on the output of the Encoder at the moment.

I could keep the existing full pipeline for the mel path (X with separators and annotations → prenet → encoder → proj_m). 
For the duration/prosody path, I could build a parallel sequence X_clean, same phonemes and annotations, no separators,
and pass it through a shared prenet + a separate duration encoder, then feed that into proj_w.
I can probably use the same Encoder with less params: 
```
dur_encoder = Encoder(
    encoder_params.n_channels + spk_emb_dim,
    encoder_params.filter_channels,
    encoder_params.n_heads,
    n_layers=2,           # shallower
    kernel_size=3,        # narrower receptive field is fine for duration
    p_dropout=encoder_params.p_dropout,
)
```
Which addresses problem 1.
For problem 2, I could add a single cross-attention layer after the mel encoder, where x (full sequence) attends to 
x_clean_encoded (duration encoder output):
```
x = prenet(x)
x = torch.cat([x, spk_emb], dim=-1)
x = encoder(x, x_mask)
x_clean = dur_encoder(x_clean, x_clean_mask)

x_cross = prosody_cross_attn(x, x_clean_encoded, x_clean_encoded, x_clean_mask)
x = self.norm_prosody(x + x_cross)

mu = self.proj_m(x) * x_mask
```

Cross-attention is just regular attention, but Q comes from one sequence and K, V from another.
```
Q = x_full   · Wq      # what the mel sequence is asking for
K = x_clean  · Wk      # what the clean sequence offers as keys
V = x_clean  · Wv      # what the clean sequence offers as values
attn_weights = softmax(Q · K^T / sqrt(d))
output = attn_weights · V
```

Each position in `x_full` (including separators) learns to ask "what pitch/energy context is relevant to me?" and retrieves 
it from `x_clean`. The separator positions will attend to nearby phonemes in the clean sequence — exactly what you want.

In code, using the `MultiHeadAttention` you already have:
```
self.prosody_cross_attn = MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout)
self.norm_prosody = LayerNorm(hidden_channels)
cross_mask = x_mask.unsqueeze(2) * x_clean_mask.unsqueeze(-1)
y = self.prosody_cross_attn(x, x_clean_encoded, cross_mask)
x = self.norm_prosody(x + y)
```

The mask shape is the only thing that changes vs self-attention — rows are `T_full`, columns are `T_clean`.