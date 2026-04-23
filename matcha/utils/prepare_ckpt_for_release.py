"""
Strips a training checkpoint down to inference-only keys (state_dict + hyper_parameters)
and saves it to docker/checkpoint.ckpt.
"""
import sys
from pathlib import Path

import torch

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

    print(f"Saving to {DESTINATION} ...")
    torch.save(stripped, DESTINATION)

    source_mb = source_path.stat().st_size / 1024 / 1024
    destination_mb = DESTINATION.stat().st_size / 1024 / 1024
    print(f"Done. {source_mb:.1f} MB -> {destination_mb:.1f} MB")


if __name__ == "__main__":
    main()
