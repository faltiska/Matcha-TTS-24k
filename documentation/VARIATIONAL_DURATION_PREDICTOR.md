# Variational Duration Predictor - improvement idea

## Motivation

The Duration Predictor (DP) currently shows a train-to-validation Mean Absolute Error (MAE) gap of ~40%,
compared to ~19% for the prior loss. Standard regularisation techniques (weight decay, dropout) are already
in use. A variational formulation, as used in the VITS model family, is a principled way to add further
regularisation while also improving the naturalness and diversity of predicted durations.

The stochastic duration predictor is motivated by *naturalness and diversity* as its primary goals.
Closing the train/validation gap is a beneficial side effect.

## How it works

Instead of predicting a single log-duration scalar per phoneme, the Duration Predictor predicts a Gaussian
distribution: a mean (μ) and a log-variance (log σ²). During training, a duration sample is drawn from that
distribution using the reparameterisation trick:

```
z = μ + σ * ε,   where ε ~ N(0, 1)
```

The sampled value `z` is what gets compared to the Monotonic Alignment Search (MAS) log-duration target,
instead of the raw predictor output.

A Kullback-Leibler (KL) divergence term is added to the loss to keep the predicted distribution close to a
standard normal prior:

```
KL = -0.5 * sum(1 + log_var - μ² - exp(log_var))
```

This is weighted by a `kl_weight` hyperparameter, typically annealed from 0 at the start of training so
the model first learns reasonable duration predictions before the KL term starts constraining the distribution.

At inference, sampling is skipped and the mean μ is used directly as the predicted log-duration. Optionally,
a small amount of noise can be injected at inference time to produce natural variation between synthesis runs
of the same text.

## Why this helps generalisation

Two mechanisms work together:

1. The model cannot memorise exact training durations. It must commit to a distribution, and the KL term
   penalises overly narrow (overconfident) distributions that would allow memorisation.
2. The noise injected by sampling during training acts as a stochastic augmentation of the duration target,
   smoothing the loss landscape around the training examples.

## What changes in the code

- The Duration Predictor output head goes from 1 channel to 2 channels (mean and log-variance).
- In `matcha_tts.py`, the duration loss computation changes: instead of comparing `logw` (the raw prediction)
  to `logw_` (the MAS target), we sample `z` from the predicted distribution and compare that.
- A KL divergence loss term is added, weighted by `kl_weight` and annealed from 0.
- A new `kl_weight` and `kl_warmup_epochs` hyperparameter is added to the experiment config.
- Inference in `cli.py` uses the mean μ directly (no sampling), so inference behaviour is unchanged by default.

## Prior art

- **VITS** (Kim et al., 2021) introduced the stochastic duration predictor for end-to-end TTS and reported
  MOS of 4.43 on LJSpeech (ground truth: 4.57).
- **VITS2** (2023) further refined the duration predictor with a normalizing-flow-based approach and reported
  modest naturalness gains as part of a broader system redesign.
- **"Should you use a probabilistic duration model in TTS? Probably!"** (arxiv 2406.05401, 2024) evaluates
  stochastic vs. deterministic duration models across four corpora and finds stochastic models consistently
  improve naturalness, especially for spontaneous speech with high duration variability.

Note: none of the above papers isolate the stochastic duration predictor as the only variable, so the
benchmark numbers reflect full system comparisons, not the duration component alone.

## Notes for implementation

- The KL weight should be annealed: start at 0 and increase gradually over `kl_warmup_epochs`. Starting
  too high too early prevents the model from learning useful duration predictions before the KL term
  forces the distribution toward the prior.
- Clip log-variance to a reasonable range (e.g., [-4, 4]) to avoid numerical instability.
- The `mae/gap_dur` metric introduced in v21 is the right signal to watch: the goal is to bring it closer
  to `mae/gap_prior`.
- Only the Duration Predictor output head and the loss computation change. The Encoder, the MAS alignment,
  and the Decoder are unaffected.
- Consider running the capacity-reduction experiment (reducing the Duration Predictor from 384 to 256
  channels) first, as it is a smaller change and directly tests whether the DP is over-parameterised. If
  the gap does not close, the variational approach is the next step.
