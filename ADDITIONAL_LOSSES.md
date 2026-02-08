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
