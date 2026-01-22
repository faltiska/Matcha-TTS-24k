""" 
Adapted from https://github.com/keithito/tacotron
Convert input text to phonemes at both training and eval time. Based on eSpeak. 
List of supported languages: English, Spanish, Portuguese, French, German, Italian, Romanian, Japanese, Hebrew
Only Hiragana or Katakana is supported for Japanese, not Kanji.
"""

import logging
import re
from pathlib import Path

import phonemizer
from nemo_text_processing.text_normalization.normalize import Normalizer

logging.basicConfig()
logger = logging.getLogger("phonemizer")
logger.setLevel(logging.ERROR) # eSpeak is very verbose

# Set cache directory for NeMo grammars
cache_dir = Path.home() / ".cache" / "nemo" / "grammars"
cache_dir.mkdir(parents=True, exist_ok=True)

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
def multilingual_phonemizer(text, language):
    phonemizer = phonemizers[language]
    if not phonemizer:
        raise Exception(f"Unsupported {language=}")
    
    # Apply NeMo normalization if available for this language
    lang_code = language.split('-')[0]  # en-us -> en, fr-fr -> fr
    if lang_code in normalizers:
        text = normalizers[lang_code].normalize(text)
    
    phonemes = phonemizer.phonemize([text])[0]
    return phonemes


if __name__ == "__main__":
    from time import time
    
    test_cases = [
        ('en', [
            "I live for live broadcasts.",
            "Dr. Jones will see you at 15:00.",
            "The price is $5.00 as of Jan 21st, 2026.",
            "Call me at 555-1234 or visit 123 Main St.",
            "The temperature is -5°C or 23°F.",
            "He scored 95% on the test.",
        ]),
        ('es', [
            "El Dr. García llegará a las 15:00.",
            "El precio es $5.00 desde el 21 de enero de 2026.",
            "La temperatura es -5°C o 23°F.",
        ]),
        ('fr', [
            "Le Dr. Dupont vous verra à 15h00.",
            "Le prix est de 5,00€ au 21 janvier 2026.",
            "La température est de -5°C ou 23°F.",
        ]),
        ('de', [
            "Dr. Müller sieht Sie um 15:00 Uhr.",
            "Der Preis beträgt 5,00€ ab dem 21. Januar 2026.",
            "Die Temperatur beträgt -5°C oder 23°F.",
        ]),
        ('pt', [
            "O Dr. Silva verá você às 15:00.",
            "O preço é R$ 5,00 desde 21 de janeiro de 2026.",
            "A temperatura é -5°C ou 23°F.",
        ]),
        ('it', [
            "Il Dr. Rossi la vedrà alle 15:00.",
            "Il prezzo è €5,00 dal 21 gennaio 2026.",
            "La temperatura è -5°C o 23°F.",
        ]),
    ]

    for lang, examples in test_cases:
        lang_key = f"{lang}-us" if lang == 'en' else f"{lang}-fr" if lang == 'fr' else lang
        print(f"=== {lang.upper()} ===")
        for text in examples:
            print(f"Original:   {text}")

            if lang in normalizers:
                start = time()
                normalized = normalizers[lang].normalize(text)
                ms = int((time() - start) * 1000)
                print(f"Normalized: {normalized} ({ms} ms)")
            else:
                normalized = text
                print(f"Nemo normalization not available for {lang=}")
            
            start = time()
            phonemes = phonemizers[lang_key].phonemize([normalized])[0]
            ms = int((time() - start) * 1000)
            print(f"Phonemized: {phonemes} ({ms} ms)")

            print()
        
        