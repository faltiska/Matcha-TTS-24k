"""
Test if eSpeak performs any text normalization before phonemization.

This script compares eSpeak output (which normalizes) with epitran output
(which doesn't normalize) to detect where eSpeak maybe normalized the text.

There will be differences, but they will probably be just in which phonemes were used.
Check each difference and listen to the audio file, see if the voice actor said what espeak outputted.

Usage:
  pip install epitran
  python -m matcha.utils.test_espeak_normalization data/corpus-small-24k/train.csv
"""

import argparse
import sys
from pathlib import Path
import logging
import phonemizer

try:
    import epitran
except ImportError:
    print("ERROR: epitran not installed. Run: pip install epitran")
    sys.exit(1)

logging.basicConfig()
logger = logging.getLogger("phonemizer")
logger.setLevel(logging.ERROR)

# Pre-instantiate phonemizers for speed
phonemizers = {}
for lang in ["en-us", "en-gb", "ro", "fr-fr", "de", "es", "pt", "it", "ja", "he"]:
    phonemizers[lang] = phonemizer.backend.EspeakBackend(
        language=lang,
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
        logger=logger,
    )

# Map corpus languages to epitran codes
epitran_map = {
    "en-us": "eng-Latn",
    "en-gb": "eng-Latn",
    "ro": "ron-Latn",
    "fr-fr": "fra-Latn",
    "de": "deu-Latn",
    "es": "spa-Latn",
    "pt": "por-Latn",
    "it": "ita-Latn",
}

epitran_instances = {}
for lang, epi_code in epitran_map.items():
    try:
        epitran_instances[lang] = epitran.Epitran(epi_code)
    except:
        pass


def parse_filelist(filelist_path: Path, split_char: str = "|"):
    with open(filelist_path, encoding="utf-8") as f:
        return [tuple(line.strip().split(split_char)) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Test eSpeak text normalization")
    parser.add_argument("filelist", help="Path to train.csv or validate.csv")
    args = parser.parse_args()

    filelist_path = Path(args.filelist)
    if not filelist_path.exists():
        print(f"ERROR: File not found: {filelist_path}")
        sys.exit(1)

    entries = parse_filelist(filelist_path)
    found_count = 0
    
    output_file = filelist_path.parent / f"{filelist_path.stem}_normalization_diffs.txt"
    
    print(f"Testing {len(entries)} entries from {filelist_path.name}...")
    print("Comparing eSpeak (normalizes) vs epitran (doesn't normalize)")
    print(f"Output will be written to: {output_file}")
    print(f"{'='*80}\n")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"eSpeak Normalization Differences for {filelist_path.name}\n")
        f.write(f"{'='*80}\n\n")
        
        for i, parts in enumerate(entries, 1):
            if i % 100 == 0:
                print(f"\rProcessed {i}/{len(entries)}... (found {found_count} differences)", end="", flush=True)
            
            if len(parts) < 4:
                continue
            
            language = parts[2]
            text = parts[3]
            
            espeak_backend = phonemizers.get(language)
            epitran_backend = epitran_instances.get(language)
            
            if not espeak_backend or not epitran_backend:
                continue
            
            espeak_phonemes = espeak_backend.phonemize([text], strip=True, njobs=1)[0]
            epitran_phonemes = epitran_backend.transliterate(text)
            
            # Remove spaces and punctuation for comparison
            espeak_clean = ''.join(c for c in espeak_phonemes if c.isalpha() or c in 'ˈˌːæɑɒɔəɛɜɪʊʌaeiouˈˌːʃʒθðŋɹʔ')
            epitran_clean = ''.join(c for c in epitran_phonemes if c.isalpha() or c in 'ˈˌːæɑɒɔəɛɜɪʊʌaeiouˈˌːʃʒθðŋɹʔ')
            
            # If lengths differ significantly, eSpeak normalized
            len_diff = abs(len(espeak_clean) - len(epitran_clean))
            if len_diff > len(epitran_clean) * 0.3:  # 30% difference
                found_count += 1
                f.write(f"Line {i}: {parts[0]}\n")
                f.write(f"  Text:    {text}\n")
                f.write(f"  eSpeak:  {espeak_phonemes}\n")
                f.write(f"  epitran: {epitran_phonemes}\n")
                f.write("\n")
    
    print(f"\n\n{'='*80}")
    print(f"Found {found_count} texts where eSpeak likely normalized")
    print(f"Results written to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

