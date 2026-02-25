"""Copy speaker embeddings for a single speaker from source checkpoint into target checkpoint."""
import argparse
import shutil
import torch

EMB_KEYS = ["spk_emb_encoder.weight", "spk_emb_duration.weight", "spk_emb_decoder.weight"]


def _migrate(ckpt):
    sd = ckpt["state_dict"]
    if "spk_emb.weight" not in sd or "spk_emb_encoder.weight" in sd:
        return
    old = sd.pop("spk_emb.weight")
    for key in EMB_KEYS:
        sd[key] = old.clone()

    # The old spk_emb.weight was at index 0 in the parameter list.
    # We now have 3 embeddings instead of 1, so insert 2 new param indices
    # and duplicate their optimizer state.
    for opt_state in ckpt.get("optimizer_states", []):
        import copy
        state = opt_state["state"]
        # Shift all existing param indices up by 2 (two new params inserted at 0 and 1)
        new_state = {}
        for idx, v in state.items():
            new_state[idx + 2 if idx > 0 else idx] = v
        # Indices 0 and 1 are the two new embeddings; copy state from old index 0
        if 0 in state:
            new_state[1] = copy.deepcopy(state[0])
            new_state[2] = copy.deepcopy(state[0])
        opt_state["state"] = new_state
        for group in opt_state["param_groups"]:
            old_params = group["params"]
            # Insert 2 new indices after the first (old spk_emb was index 0)
            new_params = []
            for p in old_params:
                new_params.append(p + 2 if p > 0 else p)
            # Add the two new embedding indices right after index 0
            if 0 in old_params:
                pos = new_params.index(0)
                new_params.insert(pos + 1, 1)
                new_params.insert(pos + 2, 2)
            group["params"] = new_params


def transplant(target_path: str, source_path: str, spk_id: int):
    backup = target_path + ".bak"
    shutil.copy2(target_path, backup)
    print(f"Backed up target to {backup}")

    target = torch.load(target_path, map_location="cpu", weights_only=False)
    source = torch.load(source_path, map_location="cpu", weights_only=False)

    target_sd = target["state_dict"]
    source_sd = source["state_dict"]

    _migrate(target)
    _migrate(source)

    for key in EMB_KEYS:
        if key not in source_sd:
            raise KeyError(f"{key} not found in source checkpoint")
        if key not in target_sd:
            raise KeyError(f"{key} not found in target checkpoint")
        target_sd[key][spk_id] = source_sd[key][spk_id].clone()
        print(f"Transplanted {key}[{spk_id}]")

    torch.save(target, target_path)
    print(f"Saved updated checkpoint to {target_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="Target checkpoint path (will be updated in-place)")
    parser.add_argument("source", help="Source checkpoint path")
    parser.add_argument("spk_id", type=int, help="Speaker ID to transplant")
    args = parser.parse_args()
    transplant(args.target, args.source, args.spk_id)
