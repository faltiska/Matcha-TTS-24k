#!/usr/bin/env bash
# Compile the Romanian eSpeak-ng dictionary from espeak/src/ro into espeak/bin/ro.
#
# Use this after editing any file in espeak/src/ro (ro_rules, ro_list, ro_manual, ...).
# If you edited ro_manual, run build_ro_extra.sh instead (it regenerates ro_extra and
# then calls this script).
#
# espeak/bin/ro is a complete, self-contained eSpeak-ng data directory for Romanian:
#   phontab phonindex phondata phondata-manifest intonations   (copied from the system install)
#   lang/roa/ro                                                (voice file, copied from the system)
#   ro_dict                                                    (compiled here)
# Point phonemizer to it with MATCHA_ESPEAK_RO_DATA=espeak/bin/ro (see matcha/text/phonemizers.py).
#
# The compiled ro_dict and the phoneme tables are tied to the eSpeak-ng version that built
# them, so the script refuses to run with a version other than ESPEAK_EXPECTED_VERSION.
# Only needs the `espeak-ng` binary; no eSpeak source checkout is required.
#
# Usage: espeak/src/compile_ro.sh

set -euo pipefail

ESPEAK_EXPECTED_VERSION="${ESPEAK_EXPECTED_VERSION:-1.51}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/ro"
BIN_DIR="$(dirname "$SCRIPT_DIR")/bin/ro"

command -v espeak-ng >/dev/null || { echo "espeak-ng not found. Install it with: sudo apt install espeak-ng" >&2; exit 1; }

# "eSpeak NG text-to-speech: 1.51  Data at: /usr/lib/x86_64-linux-gnu/espeak-ng-data"
version_line="$(espeak-ng --version)"
version="$(sed -E 's/.*text-to-speech: ([^ ]+) .*/\1/' <<<"$version_line")"
system_data="$(sed -E 's/.*Data at: (.*)$/\1/' <<<"$version_line")"

if [[ "$version" != "$ESPEAK_EXPECTED_VERSION" ]]; then
    echo "espeak-ng version is $version, expected $ESPEAK_EXPECTED_VERSION." >&2
    echo "The sources in $SRC_DIR come from the $ESPEAK_EXPECTED_VERSION tag; update them first." >&2
    exit 1
fi
[[ -f "$system_data/phontab" ]] || { echo "System eSpeak data not found at $system_data" >&2; exit 1; }

# Fresh output dir, so stale files never survive a rebuild.
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
mkdir -p "$tmp_dir/lang/roa"
cp "$system_data"/{phontab,phonindex,phondata,phondata-manifest,intonations} "$tmp_dir/"
cp "$system_data/lang/roa/ro" "$tmp_dir/lang/roa/"

# espeak-ng compiles <lang>_rules, <lang>_list, <lang>_emoji and <lang>_extra
# from the current directory and writes <lang>_dict into --path.
(cd "$SRC_DIR" && espeak-ng --path="$tmp_dir" --compile=ro)

rm -rf "$BIN_DIR"
mkdir -p "$(dirname "$BIN_DIR")"
cp -r "$tmp_dir" "$BIN_DIR"
chmod -R u+rwX,go+rX "$BIN_DIR"

echo
echo "Compiled $BIN_DIR/ro_dict ($(stat -c %s "$BIN_DIR/ro_dict") bytes, md5 $(md5sum "$BIN_DIR/ro_dict" | cut -d' ' -f1))"
for w in copii foarte lupi; do
    printf '  %-8s %s\n' "$w" "$(espeak-ng --path="$BIN_DIR" -v ro -q --ipa "$w")"
done
