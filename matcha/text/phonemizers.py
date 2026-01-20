""" 
Adapted from https://github.com/keithito/tacotron
Convert input text to phonemes at both training and eval time. Based on eSpeak. 
List of supported languages: English, Spanish, Portuguese, French, German, Italian, Romanian, Japanese, Hebrew
Only Hiragana or Katakana is supported for Japanese, not Kanji.
"""

import logging
import re

import phonemizer

logging.basicConfig()
logger = logging.getLogger("phonemizer")
logger.setLevel(logging.ERROR) # eSpeak is very verbose

# Initializing the phonemizer globally significantly improves the speed.
phonemizers = {}

phonemizers["en-us"] = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)
phonemizers["en-gb"] = phonemizer.backend.EspeakBackend(
    language="en-gb",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)
phonemizers["ro"] = phonemizer.backend.EspeakBackend(
    language="ro",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)
phonemizers["fr-fr"] = phonemizer.backend.EspeakBackend(
    language="fr-fr",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)
phonemizers["de"] = phonemizer.backend.EspeakBackend(
    language="de",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)
phonemizers["es"] = phonemizer.backend.EspeakBackend(
    language="es",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)
phonemizers["pt"] = phonemizer.backend.EspeakBackend(
    language="pt",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)
phonemizers["it"] = phonemizer.backend.EspeakBackend(
    language="it",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)
phonemizers["ja"] = phonemizer.backend.EspeakBackend(
    language="ja",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)
phonemizers["he"] = phonemizer.backend.EspeakBackend(
    language="he",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)

# eSpeak does not remove the dot after expanding some abbreviations, like "Dr.", making the TTS pause a bit.
# Search for $dot here: https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/en_list
# Happens in other languages too, https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/es_list, 
# https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/fr_list, so on. 
def multilingual_phonemizer(text, language):
    phonemizer = phonemizers[language]
    if not phonemizer:
        raise Exception(f"Unsupported {language=}")
    phonemes = phonemizer.phonemize([text], strip=True, njobs=1)[0]
    return phonemes