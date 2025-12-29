""" from https://github.com/keithito/tacotron """
from matcha.text import phonemizers
from matcha.text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}  # pylint: disable=unnecessary-comprehension


class UnknownPhonemizerException(Exception):
    pass


def to_phoneme_ids(phonemes):
    """Converts a string of IPA phonemes to a sequence of IDs corresponding to the symbols in the text.
    Args:
      phonemes: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phoneme_ids = []
    for symbol in phonemes:
        symbol_id = _symbol_to_id[symbol]
        phoneme_ids += [symbol_id]
    return phoneme_ids


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def to_phonemes(text, language):
    return phonemizers.multilingual_phonemizer(text, language=language)
