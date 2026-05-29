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


# Conditioning Dropout on the speaker embedding (disentangle pronunciation from identity)

## The problem it solves
With FiLM speaker conditioning and only ~20 speakers, the shared encoder backbone *can still memorize
per-speaker pronunciation*. FiLM multiplies content features by an embedding-derived `gamma` (`x * gamma`),
which is a content×speaker interaction repeated at every layer. With few speakers it is cheap for the
backbone to learn "embeddings near speaker N's region → render speaker N's specific pronunciation quirks",
because a speaker's idiosyncratic pronunciation is real signal that lowers training loss.

This memorization is harmless for the main model (it sounds good on the known speakers), but it is exactly
what hurts the Style Encoder: a new embedding it predicts lands in a region where pronunciation is
entangled with identity, so the new voice comes out with an accent / mispronunciations even though the
timbre is right.

## The idea
For a random fraction of training steps, replace `speaker_embedding_enc` with zeros before it enters the
encoder. When the embedding is zero, our FiLM projection (`spk_proj`) is initialized and behaves as the
identity (`gamma=1`, `beta=0`), so the encoder runs fully unconditioned on those steps. This *forces* the
backbone to produce correct, generic pronunciation with no speaker information at all.

The result: pronunciation is pushed into the unconditional backbone weights and becomes
embedding-invariant, while the embedding is left to carry only voice/timbre variation. That is what makes
"pronunciation comes for free for any speaker, known or new" literally true, and it makes the off-manifold
embeddings the Style Encoder predicts safe to use.

## Notes for implementation
- Only drop the *encoder* embedding (`speaker_embedding_enc`). The duration predictor embedding is a
  separate concern; decide separately whether to drop it too.
- Drop per-sample (independent Bernoulli per batch element), not per-batch, so each step still sees a mix.
- Start with a modest rate (e.g. 0.1) and watch per-speaker prior-loss spread and the Style Encoder's
  embedding-distance metrics.
- Inference is unaffected (always uses the real embedding). Config-gate the rate so v19 / other experiments
  are untouched.

# Let the speaker embedding tables receive weight decay (or a norm penalty)

`configure_optimizers` in `baselightningmodule.py` currently puts every `nn.Embedding` (including both
speaker tables) in the no-decay group, so the speaker embeddings get zero weight-decay pressure today.
That means nothing pulls speakers toward a shared region, which makes it easier for the backbone to carve
out isolated per-speaker modes (the memorization described above).

Allowing the speaker tables to decay — or adding a separate, tunable norm penalty just on them — pulls
speakers together and shrinks those memorized modes, complementing conditioning dropout. This is a dial,
not a switch: it is in mild tension with the "all character lives in the embedding" goal, so it needs to be
tuned rather than turned on hard. A related, lower-priority variant is a low-rank `spk_proj` (bottleneck the
conditioning path) to cap how much speaker-specific *content* FiLM can inject without reducing the
backbone's pronunciation capacity.

Note: shrinking the core backbone is *not* the right lever for genericity — it mostly costs pronunciation
quality, which is the thing the backbone must own. The levers above target identity↔content entanglement
directly instead.
