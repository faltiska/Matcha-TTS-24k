import unicodedata

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
pre_annotations = "ˈˌ"
post_annotations = "ːˑ‿ʰʱʲʷˠˤ˞ⁿˡʼʴ̩̯̃̚" # last four characters are all combining diacritics that are invisible on their own. 
all_annotations = pre_annotations + post_annotations

ipa_symbols = vowels + consonants + pre_annotations + post_annotations

base_phonemes = vowels + consonants

syllabic_consonants = "nlm"
semi_vowels = "wj"
pre_annotatable = vowels + syllabic_consonants + semi_vowels

pre_annotated = [pre + base for pre in pre_annotations for base in pre_annotatable]
post_annotated = [base + post for base in base_phonemes for post in post_annotations]
pre_post_annotated = [pre + base + post for pre in pre_annotations for base in pre_annotatable for post in post_annotations]

symbols = [_separator] + list(_punctuation) + list(ipa_symbols) + pre_annotated + post_annotated + pre_post_annotated

symbol_to_id = {s: i for i, s in enumerate(symbols)}

SPACE_ID = symbols.index(" ")


def to_phoneme_ids(phonemes):
    """Converts a string of IPA phonemes to a sequence of IDs corresponding to the symbols in the text.
    Args:
      phonemes: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    separator_id = symbol_to_id[_separator]
    ids = []
    for symbol in phonemes.split(_separator):
        ids.append(symbol_to_id.get(symbol))
        ids.append(separator_id)
    return ids[:-1]  # remove trailing separator
