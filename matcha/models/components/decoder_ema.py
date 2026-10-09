"""
Exponential moving average (EMA) of the Decoder weights.

The flow matching loss draws a new timestep and new noise for every sample, so the Decoder's gradient is very
noisy and its weights keep jittering around the minimum even late in training. The EMA copy averages roughly
the last 1 / (1 - decay) optimizer steps, which places it near the centre of that jitter. It is never trained:
after every optimizer step it moves a small fraction of the way towards the live weights.

Only the Decoder is averaged. The Text Encoder, the Duration Predictor and the speaker embeddings are left out,
because averaging speaker embeddings blurs the best speakers towards the others.

The EMA weights are saved in the checkpoint next to the live ones, under their own key prefix. Inference loads
them into the live Decoder's slot, so the inference model never needs to know that an EMA exists.
"""
import torch

# MatchaTTS compiles the Decoder, so the live weights sit under _orig_mod in the state_dict keys.
LIVE_DECODER_PREFIX = "decoder.estimator._orig_mod."
# The EMA copy is an uncompiled module, so its keys carry no _orig_mod.
EMA_DECODER_PREFIX = "decoder_ema."

# The decay ramps up from 0.1 at the first step towards the configured value, so an EMA that starts from a
# freshly initialized Decoder does not hold on to random weights for tens of thousands of steps.
# A Decoder resumed from a long run is past the ramp at once, which is right because its weights are already good.
DECAY_WARMUP_OFFSET = 10


def effective_decay(configured_decay: float, step: int) -> float:
    warmup_decay = (1 + step) / (DECAY_WARMUP_OFFSET + step)
    return min(configured_decay, warmup_decay)


@torch.no_grad()
def update_ema_weights(ema_weights: list[torch.Tensor], live_weights: list[torch.Tensor], decay: float):
    """
    Moves every EMA tensor towards its live counterpart: ema = decay * ema + (1 - decay) * live.
    A single foreach call updates all tensors in a few fused kernels, instead of hundreds of separate launches.
    """
    torch._foreach_lerp_(ema_weights, live_weights, 1.0 - decay)


def has_ema_weights(state_dict: dict) -> bool:
    return any(key.startswith(EMA_DECODER_PREFIX) for key in state_dict)


def remove_ema_weights(state_dict: dict) -> dict:
    return {key: value for key, value in state_dict.items() if not key.startswith(EMA_DECODER_PREFIX)}


def add_ema_weights_copied_from_live(state_dict: dict):
    """
    Adds EMA weights equal to the live Decoder weights, for checkpoints saved before EMA was enabled.
    The checkpoint can then be resumed with EMA on, and the average starts from the weights trained so far.
    """
    live_keys = [key for key in state_dict if key.startswith(LIVE_DECODER_PREFIX)]
    for live_key in live_keys:
        ema_key = EMA_DECODER_PREFIX + live_key[len(LIVE_DECODER_PREFIX):]
        state_dict[ema_key] = state_dict[live_key].clone()


def state_dict_with_ema_decoder(state_dict: dict) -> dict:
    """
    Returns a state dict holding only the live Decoder keys, whose values are the EMA weights.
    Every live Decoder key is overwritten, so a strict load still checks that the two copies match one to one.
    """
    decoder_weights = remove_ema_weights(state_dict)
    for key, value in state_dict.items():
        if not key.startswith(EMA_DECODER_PREFIX):
            continue
        live_key = LIVE_DECODER_PREFIX + key[len(EMA_DECODER_PREFIX):]
        if live_key not in decoder_weights:
            raise KeyError(f"EMA weight '{key}' has no live Decoder counterpart '{live_key}'")
        decoder_weights[live_key] = value
    return decoder_weights


def select_decoder_weights(state_dict: dict, use_decoder_ema: bool) -> tuple[dict, bool]:
    """
    Returns a state dict for the inference model and whether it holds the EMA Decoder weights.
    Checkpoints trained without EMA fall back to the live weights.
    """
    use_ema = use_decoder_ema and has_ema_weights(state_dict)
    if use_ema:
        return state_dict_with_ema_decoder(state_dict), True
    return remove_ema_weights(state_dict), False


@torch.no_grad()
def swap_weights(first_weights: list[torch.Tensor], second_weights: list[torch.Tensor]):
    """
    Exchanges the values of two weight sets in place. The tensors keep their identity, so the compiled Decoder,
    its guards and the optimizer state keep pointing at the same parameters.
    Autocast caches the bf16 copies of the weights it has already cast, so the cache is cleared to make the next
    forward pass read the new values.
    """
    first_values = [weight.clone() for weight in first_weights]
    torch._foreach_copy_(first_weights, second_weights)
    torch._foreach_copy_(second_weights, first_values)
    torch.clear_autocast_cache()


def paired_flow_matching_losses(cfm, ema_weights, x1, mask, mu):
    """
    Reads the flow matching loss and its late-trajectory variant for the live Decoder and for the EMA Decoder,
    on the same batch.

    The random timestep and noise move the loss far more than the difference between the two weight sets does,
    so the EMA readings must replay exactly the draws of the live readings for the two to be comparable.
    Forking the random generators runs the live readings and then rewinds them, so the EMA readings draw the
    same values.

    The EMA weights are swapped into the live Decoder for their readings, so both readings run the same compiled
    code. A separate compiled copy would add new compilation variants, and past the recompilation limit some of
    them would fall back to eager kernels whose numerics differ.
    """
    if mu.device.type == "cuda":
        cuda_devices = [mu.device]
    else:
        cuda_devices = []

    with torch.random.fork_rng(devices=cuda_devices):
        live_losses = read_flow_matching_losses(cfm, x1, mask, mu)

    live_weights = list(cfm.estimator.parameters())
    swap_weights(live_weights, ema_weights)
    try:
        ema_losses = read_flow_matching_losses(cfm, x1, mask, mu)
    finally:
        swap_weights(live_weights, ema_weights)
    return live_losses, ema_losses


def read_flow_matching_losses(cfm, x1, mask, mu):
    diff_loss = cfm.compute_loss(x1=x1, mask=mask, mu=mu)
    # A second reading of the same loss, pinned to the end of the trajectory, for validation only.
    late_diff_loss = cfm.compute_loss(x1=x1, mask=mask, mu=mu, sample_late_trajectory=True)
    return diff_loss, late_diff_loss
