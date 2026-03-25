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
from matcha.text.symbols import _punctuation, _separator, all_annotations, post_annotations, pre_annotations, symbol_to_id

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


def validate_group(group):
    """
    Checks whether combined_group is a known symbol.
    If not, the last character should start a new group instead of being appended.
    This is a safety net for phoneme combinations that split_ipa's grouping rules
    would combine, but that are not present in the symbols vocabulary.
    """
    known_group = group in symbol_to_id
    if not known_group:
        logger.warning(f"Unknown phoneme group '{group}' — will keep individual characters.")
    return known_group


def split_ipa(phonemes):
    """
    This method creates phonemes groups that contain phonemes that only make sense together:

    1. Pre-annotations like stress markers ("ˈˌ") stay with the phoneme right after them.
       Example: "əˈbaʊt" → ["ə", "ˈb", "a", "ʊ", "t"]

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
    
    We do not use the tie characters, and we do not group affricate or diphthongs, as they're two separate sounds.
    """
    phonemes = unicodedata.normalize('NFD', phonemes)
    result = []
    force_combine_next = False
    for char in phonemes:
        cat = unicodedata.category(char)
        
        is_combining = unicodedata.combining(char) > 0
        is_modifier = cat in ('Lm', 'Sk')
        is_pre_annotation = char in pre_annotations
        is_post_annotation = char in post_annotations
        is_backward_sticky = (is_combining or is_modifier or is_post_annotation) and not is_pre_annotation
        last_char_of_group = result[-1][-1] if result else ''
        last_char_is_annotation = last_char_of_group in all_annotations

        if char in _punctuation:
            result.append(char)
            force_combine_next = False
        elif last_char_is_annotation and (is_pre_annotation or is_post_annotation):
            result.append(char)
            force_combine_next = False
        elif (is_backward_sticky or force_combine_next) and result and not is_pre_annotation:
            group = result[-1] + char
            group = unicodedata.normalize('NFC', group)
            if validate_group(group):
                result[-1] = group
            else:
                result.append(char)
            force_combine_next = False            
        else:
            result.append(char)
            
        if is_pre_annotation:
            force_combine_next = True
            
    return result


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

    phonemes = phonemizer.phonemize([text])[0].rstrip()
    
    # Each phoneme sound has transitional sections at the start where the sound from the previous phoneme morphs into 
    # the sound of the new one and at the end, where the phoneme morphs into the the next one.   
    # The Encoder must be able to find the middle section where each phoneme sounds like "itself".
    # By adding separators between phonemes, we tell the Encoder there is something else there so it can 
    # model the transitions too: phoneme - transition - phoneme - transition ...
    phonemes = _separator.join(split_ipa(phonemes))

    return phonemes 

