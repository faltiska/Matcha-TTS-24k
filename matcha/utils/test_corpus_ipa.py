"""
Test corpus IPA symbols against symbols.py.

Usage:
  python -m matcha.utils.test_corpus_ipa corpus.csv
"""

import argparse
from pathlib import Path

from matcha.text import to_phonemes
from matcha.text.symbols import symbols


def parse_filelist(filelist_path: Path, split_char: str = "|"):
    with open(filelist_path, encoding="utf-8") as f:
        return [line.strip().split(split_char) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Test corpus IPA symbols against symbols.py.")
    parser.add_argument("corpus", help="Path to corpus CSV file")
    args = parser.parse_args()

    input_path = Path(args.corpus).resolve()
    entries = parse_filelist(input_path)
    total = len(entries)
    symbol_set = set(symbols)
    unknown_symbols = set()

    print(f"[test_corpus_ipa] Input: {input_path}")
    print(f"[test_corpus_ipa] Processing {total} entries...")

    for i, parts in enumerate(entries, start=1):
        if len(parts) < 4:
            print(f"[test_corpus_ipa] WARNING: Skipping malformed line {i}: {parts}")
            continue

        language=parts[2]
        text = parts[3]
        ipa = to_phonemes(text, language=language)

        for char in ipa:
            if char not in symbol_set:
                unknown_symbols.add(char)

        print(f"\r[test_corpus_ipa] {i}/{total} done.", end="", flush=True)

    if unknown_symbols:
        print(f"[test_corpus_ipa] WARNING: Found {len(unknown_symbols)} unknown symbols not in symbols.py:")
        print(f"[test_corpus_ipa] {sorted(unknown_symbols)}")
    else:
        print(f"[test_corpus_ipa] All symbols are valid.")


if __name__ == "__main__":
    main()
