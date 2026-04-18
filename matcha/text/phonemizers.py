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
from matcha.text.symbols import symbol_to_id, voiced_phoneme_ids, PRE_ID, POST_ID

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
    # The original code was adding te same separator everywhere, but that means the model had to assign different 
    # acoustic representations and different durations to the separator based on context. It's not great.
    # Also, it added separators everywhere, in between an annotation and a punctuation, or between punctuation symbols.
    # The transition sound is different based on surrounding phonemes so the right way to model this would be to
    # create new symbols one for each distinct sequence of phonemes. Since there are about 100 voiced phonemes, 
    # I would need 10K new symbols. A more realistic method would to add 2 new symbols for each voiced phoneme, which 
    # would increase the diction ary size only by 200. 
    # I could then replace each voiced phoneme P by a tuple of (Pre, P, Post), for example:
    # (pre_æ, æ, post_æ), (pre_ð, ð, post_ð), (pre_ə, ə, post_ə), ...
    # With this tokenization scheme, the model achieves the best quality I was able to get, except for one problem.
    # Each phoneme will be at least 3 frames long, so 3 frames must be as short as a short consonant can be in real life.
    # That is 5-15ms for stop consonants. It is true that listeners need the transitional sounds around that stop 
    # consonant to understand it, otherwise it passes by unnoticed so the length of the phoneme plus transitional sound
    # could be as long as 30-40ms. 
    # Still., I found that the 10ms frame length (required by Vocos at 24KHz with a hop of 256) is too long to support 
    # this tokenization scheme. But a 5ms frame length proved, in practice, to be perfect.
    # The problem was that each pair of voiced phonemes was separated by 2 symbols in my new scheme:
    #   pre1, p1, post1, pre2, p2, post2
    # and 20ms for transitional sounds is too long. And because transitions to consonants can be shorter than 10ms.

    ids = []
    debug_phonemes = []
    for phoneme in phonemes:
        phoneme_id = symbol_to_id[phoneme]
        is_voiced_phoneme = phoneme_id in voiced_phoneme_ids
        if is_voiced_phoneme:
            ids.extend([PRE_ID + phoneme_id, phoneme_id, POST_ID + phoneme_id])
            debug_phonemes.extend(['‹', phoneme, '›']) # this is just for display purposes
        else:
            ids.append(phoneme_id)
            debug_phonemes.append(phoneme)

    return ''.join(debug_phonemes), ids
