"""
Validate corpus IPA symbols against symbols.py.

Usage:
  python -m matcha.utils.validate_corpus_ipa -i configs/data/corpus-24k.yaml
"""

import argparse
from pathlib import Path
from typing import Dict, Any

from matcha.text.phonemizers import multilingual_phonemizer
from matcha.text.symbols import symbols


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(str(path))
        return dict(OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def _resolve_path(maybe_path: str) -> Path:
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def parse_filelist(filelist_path: Path, split_char: str = "|"):
    with open(filelist_path, encoding="utf-8") as f:
        return [line.strip().split(split_char) for line in f if line.strip()]


def validate_filelist(filelist_path: Path):
    entries = parse_filelist(filelist_path)
    total = len(entries)
    symbol_set = set(symbols)
    unknown_symbols = set()
    max_ipa_len = 0

    print(f"[validate_corpus_ipa] Input: {filelist_path}")
    print(f"[validate_corpus_ipa] Found {total} entries...")

    for i, parts in enumerate(entries, start=1):
        if len(parts) < 4:
            print(f"[validate_corpus_ipa] WARNING: Skipping malformed line {i}: {parts}")
            continue

        language = parts[2]
        text = parts[3]
        phonemes, symbol_ids = multilingual_phonemizer(text, language)
        max_ipa_len = max(max_ipa_len, len(symbol_ids))

        for symbol in phonemes:
            if symbol not in symbol_set and symbol not in unknown_symbols:
                unknown_symbols.add(symbol)
                print(f"\n[validate_corpus_ipa] Unknown symbol {repr(symbol)} in: {repr(text)}")

        print(f"\r[validate_corpus_ipa] {i}/{total} done.", end="", flush=True)

    print()
    print(f"[validate_corpus_ipa] Max IPA sequence length: {max_ipa_len}")
    if unknown_symbols:
        print(f"[validate_corpus_ipa] WARNING: Found {len(unknown_symbols)} unknown symbols not in symbols.py:")
        print(f"[validate_corpus_ipa] {sorted(unknown_symbols)}")
    else:
        print(f"[validate_corpus_ipa] All symbols are valid.")


def main():
    parser = argparse.ArgumentParser(description="Validate corpus IPA symbols against symbols.py.")
    parser.add_argument("-i", "--data-config", required=True, help="Path to data YAML (e.g., configs/data/corpus-24k.yaml)")
    args = parser.parse_args()

    cfg = _load_yaml_config(Path(args.data_config).resolve())
    train_filelist = _resolve_path(str(cfg["train_filelist_path"]))
    valid_filelist_path = cfg.get("valid_filelist_path", "")
    valid_filelist = _resolve_path(str(valid_filelist_path)) if valid_filelist_path else None

    validate_filelist(train_filelist)
    if valid_filelist and valid_filelist.exists():
        validate_filelist(valid_filelist)


if __name__ == "__main__":
    main()
