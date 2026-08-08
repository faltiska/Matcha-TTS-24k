"""
Create a checkpoint with a trained Text Encoder and Duration Predictor but a newly initialized Decoder.

The trained source contributes the phoneme embedding, pre-net, Transformer, mel projection, Duration
Predictor, and both speaker embedding tables. The fresh Decoder source contributes the corrected Decoder
architecture, its newly initialized weights, and the hyperparameters used for resumed training.

The source checkpoints must have identical Text Encoder and speaker embedding shapes. Decoder shapes may
differ, because Decoder tensors are never copied from the trained source.

Optimizer state is rebuilt as a fresh, empty Adaptive Moment Estimation with decoupled weight decay (AdamW)
state. Training progress is reset so Lightning resumes this checkpoint as a new run.

Usage:
    python -m matcha.utils.reset_decoder <trained_source.ckpt> <fresh_decoder_source.ckpt> -o <output.ckpt>

The output path is backed up to <output.ckpt>.bak before modification, if it already exists.
"""
import argparse
import shutil
from pathlib import Path

import torch


# MatchaTTS compiles the Text Encoder, so the real module is nested under _orig_mod in state_dict keys.
# This prefix includes the phoneme embedding, pre-net, Transformer, mel projection, and Duration Predictor.
TRAINED_COMPONENT_PREFIXES = (
    "encoder._orig_mod.",
    "speaker_embeddings_enc.",
    "speaker_embeddings_dur.",
)

# BaseLightningClass.on_load_checkpoint unconditionally reads epoch, so it must remain in the checkpoint.
TRAINER_PROGRESS_KEYS_TO_REMOVE = ("global_step", "loops")


def is_trained_component_key(key: str) -> bool:
    return key.startswith(TRAINED_COMPONENT_PREFIXES)


def assert_every_prefix_matches_keys(state_dict, checkpoint_description: str):
    """Fail if a renamed state_dict prefix would otherwise silently copy no trained tensors."""
    for prefix in TRAINED_COMPONENT_PREFIXES:
        prefix_matches_no_key = not any(key.startswith(prefix) for key in state_dict)
        if prefix_matches_no_key:
            raise ValueError(
                f"No key in the {checkpoint_description} state_dict starts with '{prefix}'. "
                "TRAINED_COMPONENT_PREFIXES is out of sync with the model's state_dict key names."
            )


def combine_state_dicts(trained_state, fresh_decoder_state):
    """Keep the fresh Decoder state and replace only trained Encoder and Duration Predictor tensors."""
    assert_every_prefix_matches_keys(trained_state, "trained source")
    assert_every_prefix_matches_keys(fresh_decoder_state, "fresh Decoder source")

    trained_keys_to_copy = [key for key in fresh_decoder_state if is_trained_component_key(key)]
    combined_state = fresh_decoder_state.copy()

    for key in trained_keys_to_copy:
        if key not in trained_state:
            raise ValueError(
                "The trained source checkpoint is missing an expected trained-component key, "
                f"so it cannot be combined safely: {key}"
            )

        trained_value = trained_state[key]
        fresh_decoder_value = fresh_decoder_state[key]
        if trained_value.shape != fresh_decoder_value.shape:
            raise ValueError(
                f"Trained-component key '{key}' has shape {tuple(trained_value.shape)} in the trained "
                f"source but {tuple(fresh_decoder_value.shape)} in the fresh Decoder source. The Text "
                "Encoder and speaker embedding configurations must be identical."
            )

        combined_state[key] = trained_value

    print(f"Copied {len(trained_keys_to_copy)} trained Encoder and Duration Predictor tensors")
    return combined_state


def build_fresh_optimizer_states(hparams):
    """Build parameter groups matching the fresh architecture, with no saved optimizer moments."""
    from matcha.models.matcha_tts import MatchaTTS

    model_hparams = dict(hparams)
    model_hparams.pop("scheduler", None)
    optimizer_partial = model_hparams.pop("optimizer")

    model = MatchaTTS(optimizer=optimizer_partial, scheduler=None, **model_hparams)
    optimizer = model.configure_optimizers()

    fresh_optimizer_state = optimizer.state_dict()
    fresh_optimizer_state["state"] = {}
    return [fresh_optimizer_state]


def reset_decoder_checkpoint(trained_source_path: str, fresh_decoder_source_path: str, output_path: str):
    """Write a new-run checkpoint with trained Encoder/Duration Predictor and fresh Decoder weights."""
    output = Path(output_path)
    if output.exists():
        backup_path = output.with_suffix(output.suffix + ".bak")
        shutil.copy2(output, backup_path)
        print(f"Backup saved to {backup_path}")

    trained_checkpoint = torch.load(trained_source_path, map_location="cpu", weights_only=False)
    fresh_decoder_checkpoint = torch.load(fresh_decoder_source_path, map_location="cpu", weights_only=False)

    combined_checkpoint = fresh_decoder_checkpoint.copy()
    combined_checkpoint["state_dict"] = combine_state_dicts(
        trained_checkpoint["state_dict"],
        fresh_decoder_checkpoint["state_dict"],
    )

    combined_checkpoint["optimizer_states"] = build_fresh_optimizer_states(
        fresh_decoder_checkpoint["hyper_parameters"]
    )
    combined_checkpoint["epoch"] = 0
    for key in TRAINER_PROGRESS_KEYS_TO_REMOVE:
        combined_checkpoint.pop(key, None)

    torch.save(combined_checkpoint, output_path)
    print(f"Saved reset checkpoint to {output_path}")
    print("Copied from trained source: Text Encoder, Duration Predictor, and both speaker embedding tables")
    print("Kept from fresh Decoder source: Decoder tensors, hyperparameters, and all other state")
    print("Optimizer state was rebuilt fresh and training progress was reset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trained_source",
        help="Checkpoint containing the trained Text Encoder and Duration Predictor",
    )
    parser.add_argument(
        "fresh_decoder_source",
        help="Fresh v22 checkpoint containing the newly initialized Decoder",
    )
    parser.add_argument("-o", "--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()
    reset_decoder_checkpoint(args.trained_source, args.fresh_decoder_source, args.output)
