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
from pathlib import Path

# Set cache directory for NeMo grammars
cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
cache_dir = cache_base / "nemo" / "grammars"
cache_dir.mkdir(parents=True, exist_ok=True)

import phonemizer
from nemo_text_processing.text_normalization.normalize import Normalizer

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
    text = re.sub('[\"„“”«»]', '', text)
    text = re.sub(r'\s*[<>()[\]{}—–…]\s*', ', ', text)
    text = re.sub(r'^,\s*', '', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r',\s*([.?!])', r'\1', text)

    text = text.rstrip()
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
    phonemizer = phonemizers[language]
    if not phonemizer:
        raise Exception(f"Unsupported {language=}")
    
    # Apply NeMo normalization if available for this language.
    # The eSpeak normalization will still be applied during phonemization; in case of languages 
    # supported by Nemo, it will probably not do anything, as it's already normalized, which for 
    # those not supported, eSpeak will take care of it.
    lang_code = language.split('-')[0]  # en-us -> en, fr-fr -> fr
    text = normalize_text(lang_code, text)

    text = cleanup_text(text)

    phonemes = phonemizer.phonemize([text])[0]

    return phonemes


if __name__ == "__main__":
    """
    Run me with:
     python -m matcha.text.phonemizers
    """
    from time import time
    
    test_cases = [
        ('en', [
            "I live for live broadcasts.",
            "Dr. Jones will see you at 15:00.",
            "The price is $5.00 as of Jan 21st, 2026.",
            "Call me at 555-1234 or visit 123 Main St.",
            "The temperature is -5°C or 23°F.",
            "He scored 95% on the test.",
            "Word   ",
            "Word\n\n",
            "Word\t",
            'He said “hello” to me and I\'ve said hello ‘back’.',
            "The value is <10> or (20) or [30] or {40}.",
            "It was a dark and stormy night—except at occasional intervals.",
            "The years 2020–2025 were challenging.",
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
            "Elle a dit «bonjour» à lui.",
            "La pluie tombait à torrents—sauf à intervalles occasionnels.",
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
        ('ro', [
            "Dr. Popescu vă va vedea la ora 15:00.",
            "Prețul este 5,00 lei din 21 ianuarie 2026.",
            "Temperatura este -5°C sau 23°F.",
            "Sunați-mă la 555-1234 sau vizitați Str. Principală nr. 123.",
            "Oare?",
            "Doare!",
            "N-are.",
            "Cuvânt   ",
            "Cuvânt\n\n",
            "Cuvânt\t",
            "Ploaia cădea în torente—cu excepția momentelor ocazionale.",
        ]),
    ]

    for lang, examples in test_cases:
        lang_key = f"{lang}-us" if lang == 'en' else f"{lang}-fr" if lang == 'fr' else lang
        print(f"=== {lang.upper()} ===")
        for text in examples:
            print(f"Original:   <{text}>")

            if lang in normalizers:
                start = time()
                normalized = normalize_text(lang, text)
                ms = int((time() - start) * 1000)
                print(f"Normalized: <{normalized}> ({ms} ms)")
            else:
                normalized = text
                print(f"Nemo normalization not available for {lang=}")
            
            cleaned = cleanup_text(normalized)
            print(f"Cleaned:    <{cleaned}>")
            
            start = time()
            phonemes = multilingual_phonemizer(text, lang_key)
            ms = int((time() - start) * 1000)
            print(f"Phonemized: <{phonemes}> ({ms} ms)")

            print()
        
        
