#!/usr/bin/env bash
# Regenerate espeak/src/ro/ro_listx from RoLEX and compile espeak/bin/ro.
#
# Run this after editing ro_rules or ro_list (so ro_listx only keeps words the rules still
# get wrong), or after bumping the RoLEX commit. For a one-word fix just edit
# espeak/src/ro/ro_extra and run espeak/src/compile_ro.sh; no RoLEX needed.
#
# Needs: espeak-ng (same version as in compile_ro.sh), the repo venv (phonemizer), curl.
# RoLEX is downloaded once to .cache/rolex (gitignored) and checked against a pinned hash.
#
# Usage: espeak/src/build_ro_listx.sh

set -euo pipefail

ROLEX_COMMIT="bcdce3ce7b06a525a82040cf91afa6e9674af769"
ROLEX_SHA256="c07b13f0bd08b44242f7d6cc7d4f44abd7175b54cd1c2c2dc82c7b79a099372d"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CACHE_DIR="${MATCHA_CACHE_DIR:-$REPO_DIR/.cache}/rolex"
ROLEX_FILE="$CACHE_DIR/rolex.v1.$ROLEX_COMMIT.txt"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="${PYTHON:-python}"
elif [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

if [[ ! -f "$ROLEX_FILE" ]]; then
    mkdir -p "$CACHE_DIR"
    echo "Downloading RoLEX $ROLEX_COMMIT ..."
    curl -sSLf -o "$ROLEX_FILE.part" \
        "https://raw.githubusercontent.com/adrianastan/rolex/$ROLEX_COMMIT/rolex.v1.txt"
    mv "$ROLEX_FILE.part" "$ROLEX_FILE"
fi
echo "$ROLEX_SHA256  $ROLEX_FILE" | sha256sum --check --quiet

"$PYTHON" "$SCRIPT_DIR/build_ro_listx.py" --rolex "$ROLEX_FILE" --rolex-commit "$ROLEX_COMMIT"
"$SCRIPT_DIR/compile_ro.sh"
