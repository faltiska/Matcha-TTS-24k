""" 
Adapted from https://github.com/keithito/tacotron
Convert input text to phonemes at both training and eval time. Based on eSpeak. 
List of supported languages: English, Spanish, Portuguese, French, German, Italian, Romanian, Japanese, Hebrew
Only Hiragana or Katakana is supported for Japanese, not Kanji.

Test with: pytest tests/test_phonemizers.py 
"""

import logging
import re
import os
import unicodedata
from pathlib import Path

# Set cache directory for NeMo grammars
cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
cache_dir = cache_base / "nemo" / "grammars"
cache_dir.mkdir(parents=True, exist_ok=True)

from nemo_text_processing.text_normalization.normalize import Normalizer
import phonemizer
from matcha.text.symbols import _separator, post_annotations, pre_annotations

logging.basicConfig()
logger = logging.getLogger("phonemizer")
logger.setLevel(logging.ERROR) # eSpeak is very verbose

# Initialize NeMo normalizers for supported languages
normalizers = {}
for lang in ['en', 'es', 'pt', 'de', 'fr', 'it']:
    try:
        normalizers[lang] = Normalizer(input_case='cased', lang=lang, cache_dir=str(cache_dir))
    except Exception as e:
        logger.warning(f"NeMo normalizer not available for {lang}: {e}")

# Initializing the phonemizer globally significantly improves the speed.
phonemizers = {}
for lang in ["en-us", "en-gb", "ro", "fr-fr", "de", "es", "pt", "it", "ja", "he"]:
    phonemizers[lang] = phonemizer.backend.EspeakBackend(
        language=lang,
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
        logger=logger,
    )

# eSpeak does not remove the dot after expanding some abbreviations, like "Dr.", making the TTS pause a bit.
# Search for $dot here: https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/en_list
# Happens in other languages too, https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/es_list, 
# https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/fr_list, so on. 


def cleanup_text(text):
    text = re.sub('[\"„“”«»¡¿]', '', text)
    text = re.sub(r'\s*[<>()[\]{}—–…]\s*', ', ', text)
    text = re.sub(r'^,\s*', '', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r',\s*([.?!])', r'\1', text)

    text = text.strip()
    if not text.endswith(('.', '?', '!')):
        text = text + '.'

    return text


def normalize_text(lang_code, text):
    # Nemo handles the smart left single quotes incorrectly, as if it was the standard single quotes
    # The smart right single quote is frequently used as a single quote, so I just need to remove the left one.
    # The right one is correctly handled by eSpeak.
    # E.g. "don’t" instead of "don't".
    text = re.sub('‘', '', text)
    if lang_code in normalizers:
        normalizer = normalizers[lang_code]
        text = normalizer.normalize(text)
    return text


def split_ipa(phonemes):
    """Splits a phoneme string into symbols, making sure it does not separate:
     1. Combining codepoints for example "ã" is represented as two IPA codepoints: "a" and " ̃",
        like in "un an" → "œ̃n ˈɑ̃" (French).  
     2. Diacritics/stress markers from the character they annotate.
        - pre-annotations (stress markers) attach to the next symbol: "about" → "əˈbaʊt"
        - post-annotations (diacritics, length) attach to the previous symbol: "Bahn" → "baːn"
    """
    phonemes = unicodedata.normalize('NFD', phonemes)
    result = []
    force_combine_next = False
    for char in phonemes:
        cat = unicodedata.category(char)
        is_combining = unicodedata.combining(char) > 0
        is_modifier = cat in ('Lm', 'Sk')
        is_tie = "DOUBLE" in unicodedata.name(char, "").upper()
        is_backward_sticky = is_combining or is_modifier or char in post_annotations
        if (is_backward_sticky or force_combine_next) and result:
            result[-1] += char
            force_combine_next = False
        else:
            result.append(char)
        if is_tie or char in pre_annotations:
            force_combine_next = True
    return result


def multilingual_phonemizer(text, language, insert_separators=True):
    phonemizer = phonemizers.get(language)
    if phonemizer is None:
        raise ValueError(f"Unsupported {language=}")
    
    # Apply NeMo normalization if available for this language.
    # The eSpeak normalization will still be applied during phonemization; in case of languages 
    # supported by Nemo, it will probably not do anything, as it's already normalized, which for 
    # those not supported, eSpeak will take care of it.
    lang_code = language.split('-')[0]  # en-us -> en, fr-fr -> fr
    text = normalize_text(lang_code, text)

    text = cleanup_text(text)

    phonemes = ' ' + phonemizer.phonemize([text])[0]

    if insert_separators:
        phonemes = _separator.join(split_ipa(phonemes))

    return phonemes 

