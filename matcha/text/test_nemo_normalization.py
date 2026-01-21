"""
Test if corpus text matches NeMo normalized text.

This script checks if the text in your corpus CSV is already normalized
or if it needs to be re-recorded with NeMo normalized text.

Usage:
  python -m matcha.text.test_nemo_normalization data/corpus-small-24k/train.csv
"""

import argparse
import sys
from pathlib import Path
from matcha.text.phonemizers import normalizers


def parse_filelist(filelist_path: Path, split_char: str = "|"):
    with open(filelist_path, encoding="utf-8") as f:
        return [tuple(line.strip().split(split_char)) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Test NeMo text normalization against corpus")
    parser.add_argument("filelist", help="Path to train.csv or validate.csv")
    args = parser.parse_args()

    filelist_path = Path(args.filelist)
    if not filelist_path.exists():
        print(f"ERROR: File not found: {filelist_path}")
        sys.exit(1)

    entries = parse_filelist(filelist_path)
    diff_count = 0
    
    output_file = filelist_path.parent / f"{filelist_path.stem}_nemo_diffs.txt"
    
    print(f"Testing {len(entries)} entries from {filelist_path.name}...")
    print(f"Output will be written to: {output_file}")
    print(f"{'='*80}\n")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"NeMo Normalization Differences for {filelist_path.name}\n")
        f.write(f"{'='*80}\n\n")
        
        for i, parts in enumerate(entries, 1):
            if i % 100 == 0:
                print(f"\rProcessed {i}/{len(entries)}... (found {diff_count} differences)", end="", flush=True)
            
            if len(parts) < 4:
                continue
            
            language = parts[2].split('-')[0]  # en-us -> en
            text = parts[3]
            
            normalizer = normalizers.get(language)
            if not normalizer:
                continue
            
            normalized = normalizer.normalize(text)
            
            if text != normalized:
                diff_count += 1
                f.write(f"Line {i}: {parts[0]}\n")
                f.write(f"  Original:   {text}\n")
                f.write(f"  Normalized: {normalized}\n")
                f.write("\n")
    
    print(f"\n\n{'='*80}")
    print(f"Found {diff_count} texts that differ from NeMo normalization")
    if diff_count > 0:
        print(f"Results written to: {output_file}")
        print("You may need to re-record these with normalized text.")
    else:
        print("All texts are already normalized!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
