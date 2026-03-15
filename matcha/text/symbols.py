"""
Defines the set of symbols used in text input to the model.
"""

# Token used for separating voiced phonemes (see to_phoneme_ids()) 
_separator = "|"

# Punctuation marks that may appear in phonemizer output.
# WARNING: do not reorder or remove — symbol IDs are baked into saved checkpoints.
# Most of these are stripped before reaching eSpeak by cleanup_text() in phonemizers.py.
# Only ;:,.!? and space actually survive to the phonemizer output in practice.
# ¡¿ are stripped by cleanup_text, so in practice only ;:,.!? and space appear in training data.
_punctuation = ';:,.!?¡¿_—…-\'"«»“”()[]/ '

# IPA symbols that might appear in the list of supported languages.
# I cannot check if they are supported by eSpeak, but it probably doesn't hurt 
# to have them here, even if they will not appear in real life. 
# English, Spanish, Portuguese, French, German, Italian, Romanian, Japanese, Hebrew

vowels = "aeiouɑɐɒæəɘɚɛɜɝɞɨɪɔøɵɤʉʊyɶœɯʏʌᵻ"
consonants = "bβcçdðfɡɢɣhɦɧħɥjɟʝkʎlɭʟɬɫɮmɱnɳɲŋɴpɸqrɹɺɾɽɻʀʁsʂʃtʈθvʋⱱwʍxχzʐʒʑʔʕʢʡʙɕɖʜɰ"
suprasegmentals = "ˈˌːˑ‿"
pitch_markers = "↓↑→↗↘˥˦˧˨˩"
diacritics = "ʰʱʲʷˠˤ˞ⁿˡʼʴ̩̯̃̚"
voiced_phonemes = set(vowels + consonants)


def to_phoneme_ids(phonemes):
    """Converts a string of IPA phonemes to a sequence of IDs corresponding to the symbols in the text.
    Args:
      phonemes: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    return [symbol_to_id[symbol] for symbol in phonemes]


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        result += id_to_symbol[symbol_id]
    return result
ipa_symbols = vowels + consonants + suprasegmentals + pitch_markers + diacritics

# Export all symbols:
symbols = [_separator] + list(_punctuation) + list(ipa_symbols)

symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}
