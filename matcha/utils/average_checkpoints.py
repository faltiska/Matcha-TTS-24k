"""Average weights from multiple checkpoints into a new one."""
import argparse
import torch


def average_checkpoints(paths: list[str], output: str):
    checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    avg = checkpoints[0].copy()
    state = avg["state_dict"]

    for ckpt in checkpoints[1:]:
        for k, v in ckpt["state_dict"].items():
            state[k] = state[k] + v

    for k in state:
        state[k] = state[k] / len(checkpoints)

    torch.save(avg, output)
    print(f"Saved averaged checkpoint to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="Input checkpoint paths")
    parser.add_argument("-o", "--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()
    average_checkpoints(args.checkpoints, args.output)
