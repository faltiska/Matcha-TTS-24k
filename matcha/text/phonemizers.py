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
from matcha.text.symbols import separator, symbols, symbol_to_id

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
    text = re.sub(r'\s*[,<>()[\]{}—–…]\s*', ', ', text)
    text = re.sub(r'\s+([.?!,;:])', r'\1', text) # no spaces before punctuation
    text = re.sub(r'^,\s*', '', text) # no leading comma
    text = re.sub(r',\s*,', ',', text) # no multiple commas
    text = re.sub(r',\s*([.?!])', r'\1', text) # no comma before sentence punctuation 

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


def multilingual_phonemizer(text, language):
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

    # There's always a bit of silence at the start, so I am adding a space for it.
    # I don't need a space at the end, because there is always punctuation at the end of the sentence.
    phonemes = " " + phonemizer.phonemize([text])[0].rstrip()
    
    # Each phoneme sound has transitional sections at the start where the sound from the previous phoneme morphs into 
    # the sound of the new one and at the end, where the phoneme morphs into the next one.   
    # The Encoder must be able to find the middle section where each phoneme sounds like "itself".
    # By adding separators between phonemes, we tell the Encoder there is something else there so it can 
    # model the transitions too: phoneme - transition - phoneme - transition ...

    phonemes = separator.join(phonemes)
    ids = [symbol_to_id[phoneme] for phoneme in phonemes]

    return phonemes, ids


def phone_id_to_display(phone_id):
    return symbols[phone_id]