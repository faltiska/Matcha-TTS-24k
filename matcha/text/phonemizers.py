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
from matcha.text.symbols import SEPARATOR_ID, annotation_ids, symbol_to_id, symbols, voiced_phoneme_ids

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

    # I tried to add some padding symbols to match the silence in the audio files, but it broke duration predciton. 
    # I had 200 ms leading silence and 800ms trailing, to which I matched 4 leading spaces and 16 trailing spaces.
    # It broke duration prediction, I could not fix it no matter what. Then I tried with a different symbol instead  
    # of space, but did not help. Then I tried with 2 leading symbols, 8 trailing, but it did not help either. 
    # I finally went back this version, that has a single leading space. This fixed duration prediction.
    # I also changed the audio silence a bit, 100ms leading, 400ms trailing, but I am confident the leading / trailing
    # symbols is what fixed it, not the amount of silence, because that is what I had in v19, when worked fine. 
    phonemes = " " + phonemizer.phonemize([text])[0].rstrip()

    # Each phoneme sound has transitional sections at the start, where the previous sound morphs into it, and at the
    # end, where it morphs into the next one. The Encoder must find the middle section where each phoneme sounds
    # like "itself", and it needs a symbol to which it can give each transition, otherwise the transitions are
    # smeared over the phonemes. Separators are those symbols: phoneme - transition - phoneme - transition ...
    #
    # Separators are placed by these rules:
    # 1. Everything eSpeak outputs is kept as is.
    # 2. A single separator symbol is used everywhere.
    # 3. A separator goes between two neighbouring symbols only when at least one of them is a voiced phoneme
    #    (any vowel or consonant) and neither of them is an annotation (stress, length, nasal, palatalization, ...).
    # The reasons:
    # 1. Separators are only needed to model transitions: between two phonemes, and between a phoneme and silence.
    #    Next to word gaps and punctuation they hold the fade-out of a word and the onset of the next one.
    # 2. The Encoder can use annotation symbols to model transitions, so extra separators aren't needed next to them.
    # 3. The Encoder can learn both an annotation's intended function and the sound it holds.
    # A single separator symbol is enough because the Encoder sees the neighbours of every symbol, so its output for
    # a separator already depends on the transition it sits in.
    #
    # The alignment gives every symbol at least one mel frame, so every separator adds to the shortest possible
    # length of a word. Placing exactly one transition symbol between two phonemes keeps short consonants short.
    # Example: eSpeak's " ɪn sˌʌm fˈɔːm." becomes " |ɪ|n| |sˌʌ|m| |fˈɔːm|."
    espeak_ids = [symbol_to_id[phoneme] for phoneme in phonemes]
    ids = insert_separators(espeak_ids)
    displayed_phonemes = "".join(symbols[phoneme_id] for phoneme_id in ids)

    return displayed_phonemes, ids


def needs_separator_between(left_id, right_id):
    touches_a_voiced_phoneme = left_id in voiced_phoneme_ids or right_id in voiced_phoneme_ids
    touches_an_annotation = left_id in annotation_ids or right_id in annotation_ids
    return touches_a_voiced_phoneme and not touches_an_annotation


def insert_separators(espeak_ids):
    """Returns the eSpeak symbol ids, unchanged and in order, with separators inserted by the rules above."""
    ids = []
    for position, phoneme_id in enumerate(espeak_ids):
        is_first_symbol = position == 0
        if not is_first_symbol and needs_separator_between(espeak_ids[position - 1], phoneme_id):
            ids.append(SEPARATOR_ID)
        ids.append(phoneme_id)
    return ids
