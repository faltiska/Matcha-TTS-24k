#!/bin/bash
# Prepare a corpus for training:
#   1. Validate IPA symbols in train and validate filelists
#   2. Normalize trailing silence in wav files
#   3. Precompute mel spectrograms
#
# Usage: ./prepare_corpus.sh configs/data/corpus-24k.yaml

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <data-config.yaml>"
    exit 1
fi

DATA_CONFIG="$1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEL_DIR=$(python -c "
import yaml
from pathlib import Path
with open('$DATA_CONFIG') as f:
    cfg = yaml.safe_load(f)
mel_dir = Path(cfg['mel_dir'])
if not mel_dir.is_absolute():
    mel_dir = Path('$SCRIPT_DIR') / mel_dir
print(mel_dir.resolve())
")

read -r -p "Delete existing mel files in '$MEL_DIR'? [y/N] " DELETE_MELS
if [[ "$DELETE_MELS" =~ ^[Yy]$ ]]; then
    echo "Deleting mel files in $MEL_DIR..."
    find "$MEL_DIR" -name "*.npy" -delete
    echo "Done."
fi

echo "=== Step 1: Validate IPA symbols ==="
python -m matcha.utils.validate_corpus_ipa -i "$DATA_CONFIG"

echo "=== Step 2: Normalize trailing silence ==="
python -m matcha.utils.normalize_silence -i "$DATA_CONFIG" --target_silence 0.8

echo "=== Step 3: Precompute mel spectrograms ==="
python -m matcha.utils.precompute_mels -i "$DATA_CONFIG"

echo "=== Done ==="
