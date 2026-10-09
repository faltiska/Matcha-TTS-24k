"""
Strips a training checkpoint down to inference-only keys (state_dict + hyper_parameters)
and saves it to docker/checkpoint.ckpt. EMA Decoder weights, when present, replace the live Decoder weights.
"""
import sys
from pathlib import Path

import torch

from matcha.models.components.decoder_ema import select_decoder_weights

INFERENCE_KEYS = {"state_dict", "hyper_parameters"}
DESTINATION = Path("docker/checkpoint.ckpt")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m matcha.utils.strip_checkpoint <checkpoint.ckpt>")
        sys.exit(1)

    source_path = Path(sys.argv[1])
    if not source_path.exists():
        print(f"Error: {source_path} not found")
        sys.exit(1)

    if DESTINATION.exists():
        answer = input(f"{DESTINATION} already exists. Overwrite? [y/N] ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    print(f"Loading {source_path} ...")
    ckpt = torch.load(source_path, map_location="cpu", weights_only=False)

    stripped = {key: ckpt[key] for key in INFERENCE_KEYS if key in ckpt}

    # The released model is the one inference would load, so the EMA Decoder weights replace the live ones
    # when the checkpoint has them. The released file then holds a single Decoder and loads like any other.
    stripped["state_dict"], uses_decoder_ema = select_decoder_weights(stripped["state_dict"], use_decoder_ema=True)
    if uses_decoder_ema:
        print("Using the EMA Decoder weights.")
    else:
        print("Checkpoint has no EMA Decoder weights, using the live ones.")

    print(f"Saving to {DESTINATION} ...")
    torch.save(stripped, DESTINATION)

    source_mb = source_path.stat().st_size / 1024 / 1024
    destination_mb = DESTINATION.stat().st_size / 1024 / 1024
    print(f"Done. {source_mb:.1f} MB -> {destination_mb:.1f} MB")


if __name__ == "__main__":
    main()
