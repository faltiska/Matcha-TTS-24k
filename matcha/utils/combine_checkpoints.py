"""
Combine two checkpoints to isolate the Encoder from the moving-target problem described in v21/v22:
the Duration Predictor and the Decoder were trained against a constantly changing Encoder output for
1300+ epochs, so their validation loss stopped improving even though their training loss kept dropping.

This script takes the fully trained Encoder (emb, prenet, encoder, proj_m, speaker_embeddings_enc) from
a late checkpoint, and combines it with the near-scratch Duration Predictor, Decoder, and
speaker_embeddings_dur from an early checkpoint of the same run. The result is a checkpoint where the
Duration Predictor and Decoder can be trained from (near) scratch, against a fixed Encoder output,
instead of a moving one.

Both checkpoints must come from the same run (same architecture and hyperparameters), so that the
state_dict keys and the optimizer's positional parameter ordering line up between the two.

Usage:
    python -m matcha.utils.combine_checkpoints <encoder_source.ckpt> <dur_decoder_source.ckpt> -o <output.ckpt>

The output path is backed up to <output.ckpt>.bak before modification, if it already exists.
"""
import argparse
import shutil
from pathlib import Path

import torch

# Keys with these prefixes are taken from the Duration Predictor / Decoder source checkpoint.
# Everything else (Encoder: emb, prenet, encoder, proj_m, speaker_embeddings_enc) is kept from the
# Encoder source checkpoint.
DUR_DECODER_PREFIXES = (
    "encoder.proj_w.",
    "speaker_embeddings_dur.",
    "decoder.",
)


def is_dur_decoder_key(key: str) -> bool:
    return key.startswith(DUR_DECODER_PREFIXES)


def combine_state_dicts(encoder_state, dur_decoder_state):
    if encoder_state.keys() != dur_decoder_state.keys():
        only_in_encoder = encoder_state.keys() - dur_decoder_state.keys()
        only_in_dur_decoder = dur_decoder_state.keys() - encoder_state.keys()
        raise ValueError(
            "Checkpoints have different state_dict keys, they are not from the same run.\n"
            f"Only in encoder source: {sorted(only_in_encoder)}\n"
            f"Only in dur/decoder source: {sorted(only_in_dur_decoder)}"
        )

    combined = {}
    for key, encoder_value in encoder_state.items():
        if is_dur_decoder_key(key):
            combined[key] = dur_decoder_state[key]
        else:
            combined[key] = encoder_value
    return combined


def build_optimizer_param_names(hparams):
    """
    Reconstructs the name each optimizer param id refers to, by asking the real model.

    configure_optimizers() (see matcha/models/baselightningmodule.py) does not hand the optimizer
    parameters in state_dict()/named_parameters() order. It first splits every parameter into a
    "decay" group and a "no-decay" group (biases, Embedding weights, LayerNorm weights), in that
    order, and only then hands both groups to the optimizer as two param_groups. AdamW assigns flat,
    sequential param ids to its tracked tensors in the order it receives them, so id 0 is the first
    decay param, ids continue through all decay params, then continue (not restart) through all
    no-decay params.

    Rather than re-deriving that split here (which would silently drift from baselightningmodule.py if
    the rule ever changes), we instantiate the real model from the checkpoint's own hyperparameters,
    call its actual configure_optimizers(), and read the resulting parameter order directly from the
    optimizer's own param_groups.
    """
    from matcha.models.matcha_tts import MatchaTTS

    model_hparams = dict(hparams)
    model_hparams.pop("scheduler", None)
    optimizer_partial = model_hparams.pop("optimizer")

    model = MatchaTTS(optimizer=optimizer_partial, scheduler=None, **model_hparams)
    optimizer = model.configure_optimizers()

    param_to_name = {param: name for name, param in model.named_parameters()}
    param_names = []
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            param_names.append(param_to_name[param])
    return param_names


def combine_optimizer_states(encoder_optimizer_states, dur_decoder_optimizer_states, hparams):
    """
    Optimizer state (exp_avg, exp_avg_sq, step for AdamW) is indexed positionally by each parameter's
    id, assigned in the order configure_optimizers() hands parameters to the optimizer (decay group,
    then no-decay group) -- not in state_dict() order. See build_optimizer_param_names().

    Both checkpoints must come from the same run (same architecture and no-decay rule), which
    combine_state_dicts already confirmed by requiring identical state_dict keys; here we additionally
    require the two optimizer states to track the same set of param ids, so the positional indices are
    guaranteed to line up.
    """
    param_names = build_optimizer_param_names(hparams)
    dur_decoder_param_ids = {
        param_id for param_id, name in enumerate(param_names) if is_dur_decoder_key(name)
    }

    combined_optimizer_states = []
    for encoder_opt_state, dur_decoder_opt_state in zip(encoder_optimizer_states, dur_decoder_optimizer_states):
        if encoder_opt_state["state"].keys() != dur_decoder_opt_state["state"].keys():
            raise ValueError("Optimizer states track different parameter ids, checkpoints are not from the same run.")

        combined_state = {}
        for param_id, encoder_param_state in encoder_opt_state["state"].items():
            if param_id in dur_decoder_param_ids:
                combined_state[param_id] = dur_decoder_opt_state["state"][param_id]
            else:
                combined_state[param_id] = encoder_param_state

        combined_opt_state = encoder_opt_state.copy()
        combined_opt_state["state"] = combined_state
        combined_optimizer_states.append(combined_opt_state)

    return combined_optimizer_states


def combine_checkpoints(encoder_source_path: str, dur_decoder_source_path: str, output_path: str):
    output = Path(output_path)
    if output.exists():
        backup_path = output.with_suffix(output.suffix + ".bak")
        shutil.copy2(output, backup_path)
        print(f"Backup saved to {backup_path}")

    encoder_ckpt = torch.load(encoder_source_path, map_location="cpu", weights_only=False)
    dur_decoder_ckpt = torch.load(dur_decoder_source_path, map_location="cpu", weights_only=False)

    combined_ckpt = encoder_ckpt.copy()
    combined_ckpt["state_dict"] = combine_state_dicts(encoder_ckpt["state_dict"], dur_decoder_ckpt["state_dict"])

    combined_ckpt["optimizer_states"] = combine_optimizer_states(
        encoder_ckpt["optimizer_states"], dur_decoder_ckpt["optimizer_states"], encoder_ckpt["hyper_parameters"]
    )

    torch.save(combined_ckpt, output_path)
    print(f"Saved combined checkpoint to {output_path}")
    print("Kept from encoder source: emb, prenet, encoder, proj_m, speaker_embeddings_enc")
    print("Kept from dur/decoder source: proj_w (Duration Predictor), decoder, speaker_embeddings_dur")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder_source", help="Checkpoint with the fully trained Encoder")
    parser.add_argument("dur_decoder_source", help="Checkpoint with the near-scratch Duration Predictor and Decoder")
    parser.add_argument("-o", "--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()
    combine_checkpoints(args.encoder_source, args.dur_decoder_source, args.output)
