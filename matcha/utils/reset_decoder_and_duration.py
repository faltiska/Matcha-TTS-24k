"""
Combine two checkpoints to isolate the Encoder from the moving-target problem described in v21/v22:
the Duration Predictor and the Decoder were trained against a constantly changing Encoder output for
1300+ epochs, so their validation loss stopped improving even though their training loss kept dropping.

This script takes the near-scratch Duration Predictor, Decoder, and speaker_embeddings_dur checkpoint
(the Duration Predictor / Decoder source) as-is, then splices in the fully trained Encoder (emb,
prenet, encoder, proj_m, speaker_embeddings_enc) from a separate checkpoint (the Encoder source). The
result is a checkpoint where the Duration Predictor and Decoder can be trained from (near) scratch,
against a fixed Encoder output, instead of a moving one.

The Duration Predictor / Decoder source checkpoint is taken as-is: its state_dict is not required to
share any keys or shapes with the Encoder source, so its Duration Predictor and Decoder can use a
completely different config than the one the Encoder source was trained with (e.g. a smaller Duration
Predictor or a deeper Decoder). Only the Encoder source's Encoder-owned keys are identified and
spliced in; everything else in the Duration Predictor / Decoder source is left untouched.

Because the two checkpoints may disagree on parameter shapes and counts outside the Encoder, the
optimizer state cannot be carried over from either source: AdamW's exp_avg/exp_avg_sq buffers are
shape-matched to specific parameters, and the positional param ids used to index optimizer state are
architecture-dependent.

Lightning's checkpoint loading (see restore_optimizers_and_schedulers in
lightning/pytorch/trainer/connectors/checkpoint_connector.py) requires optimizer_states and
lr_schedulers to be present whenever ckpt_path is passed to trainer.fit, even to resume as a "fresh"
run; it does not offer a weights-only resume path. So instead of copying either source's optimizer
state, this script builds a brand new, empty-state AdamW optimizer_states entry from the combined
checkpoint's own hyperparameters (matching the Duration Predictor / Decoder source's architecture): the
param_groups (decay/no-decay split, param ids, lr, weight_decay) are real, but "state" is empty, so
AdamW lazily initializes exp_avg/exp_avg_sq on the first optimizer step, exactly as it would for a
brand new run. epoch/global_step/loops are reset to their start-of-training values for the same
reason: resuming from this checkpoint should look like starting a new run, not continuing v21's.

Usage:
    python -m matcha.utils.reset_decoder_and_duration <encoder_source.ckpt> <dur_decoder_source.ckpt> -o <output.ckpt>

The output path is backed up to <output.ckpt>.bak before modification, if it already exists.
"""
import argparse
import shutil
from pathlib import Path

import torch

# Keys with these prefixes are the Encoder's own weights (emb, prenet, encoder, proj_m,
# speaker_embeddings_enc). These are spliced into the Duration Predictor / Decoder source checkpoint
# from the Encoder source checkpoint; everything else is kept as-is from the Duration Predictor /
# Decoder source, whatever its config happens to be.
# The "_orig_mod" segment is there because MatchaTTS wraps self.encoder in torch.compile() (see
# matcha/models/matcha_tts.py), which nests the real module under that attribute in the state_dict.
# Note that "encoder._orig_mod.proj_w." (the Duration Predictor) is deliberately absent: it lives
# inside the Encoder module but is one of the components being reset near-scratch.
ENCODER_PREFIXES = (
    "encoder._orig_mod.emb.",
    "encoder._orig_mod.prenet.",
    "encoder._orig_mod.encoder.",
    "encoder._orig_mod.proj_m.",
    "speaker_embeddings_enc.",
)


def is_encoder_key(key: str) -> bool:
    return key.startswith(ENCODER_PREFIXES)


def assert_every_prefix_matches_keys(state_dict, checkpoint_description: str):
    """
    Guards against ENCODER_PREFIXES silently drifting out of sync with the real state_dict key names,
    which is exactly how a previous version of this script produced a broken checkpoint: the prefixes
    were missing the torch.compile() "_orig_mod" segment, so they matched no keys at all, the trained
    Encoder was never spliced in, and nothing failed because "no keys matched" looked identical to
    "nothing was missing".
    """
    for prefix in ENCODER_PREFIXES:
        prefix_matches_no_key = not any(key.startswith(prefix) for key in state_dict)
        if prefix_matches_no_key:
            raise ValueError(
                f"No key in the {checkpoint_description} state_dict starts with '{prefix}'. "
                "ENCODER_PREFIXES is out of sync with the model's state_dict key names."
            )


def combine_state_dicts(encoder_state, dur_decoder_state):
    """
    Takes dur_decoder_state as-is, then overwrites only the Encoder-owned keys with encoder_state's
    values. dur_decoder_state's Duration Predictor / Decoder / speaker_embeddings_dur keys are never
    inspected, so they are free to use a different config (different shapes, different key names)
    than whatever the Encoder source was trained with.
    """
    assert_every_prefix_matches_keys(encoder_state, "encoder source")
    assert_every_prefix_matches_keys(dur_decoder_state, "dur/decoder source")

    encoder_keys_to_splice = [key for key in dur_decoder_state if is_encoder_key(key)]
    combined = dur_decoder_state.copy()
    for key in encoder_keys_to_splice:
        if key not in encoder_state:
            raise ValueError(
                "Encoder source checkpoint is missing an expected Encoder key, it may not be a "
                f"matching checkpoint: {key}"
            )

        encoder_value = encoder_state[key]
        dur_decoder_value = dur_decoder_state[key]
        if encoder_value.shape != dur_decoder_value.shape:
            raise ValueError(
                f"Encoder key '{key}' has shape {tuple(encoder_value.shape)} in the encoder source but "
                f"{tuple(dur_decoder_value.shape)} in the dur/decoder source. The Encoder config must be "
                "identical in both checkpoints, since the whole point is to reuse the trained Encoder."
            )
        combined[key] = encoder_value

    print(f"Spliced {len(encoder_keys_to_splice)} Encoder tensors from the encoder source")
    return combined


# Trainer-progress keys reset to their start-of-training values. Since the combined checkpoint's
# optimizer_states is a brand new, empty-state optimizer (see build_fresh_optimizer_states), resuming
# from this checkpoint should look like starting a new run, not continuing wherever either source
# checkpoint left off.
# "epoch" is reset to 0 rather than removed: BaseLightningModule.on_load_checkpoint (see
# matcha/models/baselightningmodule.py) unconditionally reads checkpoint["epoch"], so the key must
# still be present.
TRAINER_PROGRESS_KEYS_TO_REMOVE = ("global_step", "loops")


def build_fresh_optimizer_states(hparams):
    """
    Builds a brand new, empty-state AdamW optimizer_states entry from the given model hyperparameters,
    matching the real param_groups (decay/no-decay split, param ids, lr, weight_decay) that
    configure_optimizers() (see matcha/models/baselightningmodule.py) would produce for a model with
    this architecture, but with an empty "state" dict so AdamW lazily initializes exp_avg/exp_avg_sq on
    the first optimizer step, exactly as it would for a brand new run.

    This is instantiated from the real model rather than hand-rolled here, so it can't silently drift
    from configure_optimizers() if the decay/no-decay rule or optimizer type ever changes.
    """
    from matcha.models.matcha_tts import MatchaTTS

    model_hparams = dict(hparams)
    model_hparams.pop("scheduler", None)
    optimizer_partial = model_hparams.pop("optimizer")

    model = MatchaTTS(optimizer=optimizer_partial, scheduler=None, **model_hparams)
    optimizer = model.configure_optimizers()

    fresh_state = optimizer.state_dict()
    fresh_state["state"] = {}
    return [fresh_state]


def combine_checkpoints(encoder_source_path: str, dur_decoder_source_path: str, output_path: str):
    output = Path(output_path)
    if output.exists():
        backup_path = output.with_suffix(output.suffix + ".bak")
        shutil.copy2(output, backup_path)
        print(f"Backup saved to {backup_path}")

    encoder_ckpt = torch.load(encoder_source_path, map_location="cpu", weights_only=False)
    dur_decoder_ckpt = torch.load(dur_decoder_source_path, map_location="cpu", weights_only=False)

    # The Duration Predictor / Decoder source is taken as-is (its hyper_parameters describe the
    # combined checkpoint's architecture); only its Encoder-owned keys get overwritten below.
    combined_ckpt = dur_decoder_ckpt.copy()
    combined_ckpt["state_dict"] = combine_state_dicts(encoder_ckpt["state_dict"], dur_decoder_ckpt["state_dict"])

    # Neither source's optimizer state is carried over: the Duration Predictor / Decoder source
    # checkpoint may have a different architecture than the Encoder source (that's the point of
    # resetting them near-scratch), so their optimizer state's shapes and positional param ids cannot
    # be assumed to line up. A brand new, empty-state optimizer is built instead, matching the
    # Duration Predictor / Decoder source's own architecture (the one being kept for those components).
    combined_ckpt["optimizer_states"] = build_fresh_optimizer_states(dur_decoder_ckpt["hyper_parameters"])
    combined_ckpt["epoch"] = 0
    for key in TRAINER_PROGRESS_KEYS_TO_REMOVE:
        combined_ckpt.pop(key, None)

    torch.save(combined_ckpt, output_path)
    print(f"Saved combined checkpoint to {output_path}")
    print("Kept from encoder source: emb, prenet, encoder, proj_m, speaker_embeddings_enc")
    print("Kept as-is from dur/decoder source: everything else (proj_w, decoder, speaker_embeddings_dur, "
          "and hyper_parameters), whatever its config")
    print("Optimizer state was rebuilt fresh (empty), not combined. epoch/global_step/loops were reset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder_source", help="Checkpoint with the fully trained Encoder")
    parser.add_argument("dur_decoder_source", help="Checkpoint with the near-scratch Duration Predictor and Decoder")
    parser.add_argument("-o", "--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()
    combine_checkpoints(args.encoder_source, args.dur_decoder_source, args.output)
