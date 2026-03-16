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
from matcha.text.symbols import _punctuation, _separator, all_annotations, post_annotations, pre_annotations

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
    """
    Splits an IPA string into a list of phoneme symbols, keeping multi-codepoint
    characters together as a single symbol.

    The string is first NFD-decomposed so that composed characters (e.g. "ã") are
    broken into a base letter plus combining codepoints. After splitting, each
    symbol is re-composed to NFC before being returned.

    This method creates phonemes groups that contain phonemes that only make sense together:

    1. Pre-annotations like stress markers ("ˈˌ") stay with the phoneme right after them.
       Example: "əˈbaʊt" → ["ə", "ˈb", "aʊ", "t"]

    2. Post-annotations like diacritics and length marks ("ː,") stay with the phoneme right before them.
       Example: "baːn" → ["b", "aː", "n"]

    3. Unicode combining codepoints are kept together with the character they modify.
       These appear after NFD decomposition splits a composed character into its base letter
       plus a combining codepoint. For example, the nasal vowel "ɑ̃" (French "an") is
       decomposed into "ɑ" + combining tilde "̃", which must be kept together.
       Example: "œ̃n ˈɑ̃" → ["œ̃", "n", " ", "ˈɑ̃"]

    4. Modifier letters (Unicode spacing modifier letters not covered by post-annotations)
       stay with the phoneme they modify, for the same reason as combining codepoints.

    5. Punctuation should not be part of a group.

    6. Two consecutive annotations should not be part of a group.

    7. Affricate tie characters keep the two phonemes they bridge as one symbol.
       An affricate is a single sound that begins as a stop and releases as a fricative,
       like the "ch" in "church" ("t͡ʃ") or the "j" in "judge" ("d͡ʒ").
    """
    phonemes = unicodedata.normalize('NFD', phonemes)
    result = []
    force_combine_next = False
    for char in phonemes:
        cat = unicodedata.category(char)
        is_combining = unicodedata.combining(char) > 0
        is_modifier = cat in ('Lm', 'Sk')
        is_tie = "DOUBLE" in unicodedata.name(char, "").upper()
        is_backward_sticky = (is_combining or is_modifier or char in post_annotations) and char not in pre_annotations
        is_annotation = char in pre_annotations or char in post_annotations
        last_char_of_group = result[-1][-1] if result else ''
        last_char_is_annotation = last_char_of_group in all_annotations
        if char in _punctuation:
            result.append(char)
            force_combine_next = False
        elif last_char_is_annotation and is_annotation:
            result.append(char)
            force_combine_next = False
        elif (is_backward_sticky or force_combine_next) and result:
            result[-1] += char
            force_combine_next = False            
        else:
            result.append(char)
        if is_tie or char in pre_annotations:
            force_combine_next = True
    return [unicodedata.normalize('NFC', symbol) for symbol in result]


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

    phonemes = ' ' + phonemizer.phonemize([text])[0]
    
    # Each phoneme sound has transitional sections at the start where the sound from the previous phoneme morphs into 
    # the sound of the new one and at the end, where the phoneme morphs into the the next one.   
    # The Encoder must be able to find the middle section where each phoneme sounds like "itself".
    # By adding separators between phonemes, we tell the Encoder there is something else there so it can 
    # model the transitions too: phoneme - transition - phoneme - transition ...
    phonemes = _separator.join(split_ipa(phonemes))

    return phonemes 

