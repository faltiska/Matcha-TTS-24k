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

### The Sway Sampling Logic

The core idea is to transform a linear time sequence using a power function. For Matcha, we typically want more density near  (the noise) to ensure the initial direction is correct.

### Python Implementation

Here is a concise implementation of a Sway Scheduler. You can use this to generate the `t` values that you pass into your Matcha `forward` or `inference` calls.

```python
import torch

def get_sway_schedule(num_steps, sway_coefficient=1.5, device="cpu"):
    """
    Generates a non-linear time schedule for Flow Matching.
    
    Args:
        num_steps: Number of inference steps (e.g., 10 or 50).
        sway_coefficient: Higher values (> 1.0) push more steps 
                          toward the noise (t=0).
    """
    # 1. Create linear steps from 0 to 1
    # We use num_steps + 1 because we want the intervals between them
    steps = torch.linspace(0, 1, num_steps + 1, device=device)
    
    # 2. Apply the Sway transformation
    # This formula 'sways' the density of the points
    sway_steps = steps.pow(sway_coefficient)
    
    # 3. Flip it if you are moving from 1 -> 0 (standard for some ODE solvers)
    # Matcha typically flows 0 -> 1, so we keep it as is.
    return sway_steps

# Example Usage:
steps = 10
schedule = get_sway_schedule(steps, sway_coefficient=1.2)

print(f"Linear: {torch.linspace(0, 1, steps+1)}")
print(f"Sway:   {schedule}")

```

### Why This Works for Matcha

In Flow Matching, the very first step from pure noise is the most important.
If the model picks the wrong "direction," the rest of the steps are spent trying to correct it. Sway sampling ensures that first step is tiny and precise.
By slowing down at the start, you reduce the "buzzy" artifacts often found in fast-inference TTS models.

### Integration Tip

When using this with the Matcha-TTS repository, you would replace the default `torch.linspace(0, 1, steps)` 
inside the `inference` method of the `MatchaTTS` class with this `sway_steps` vector.


## Code changes
Remove app.py
Do I have to return so much data from inference?