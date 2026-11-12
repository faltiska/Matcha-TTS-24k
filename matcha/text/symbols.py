import unicodedata

"""
Defines the set of symbols used in text input to the model.
"""

# Token used for separating phonemes or phoneme groups 
separator = "|"

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
# last five characters are combining diacritics (invisible on their own): syllabic, inverted breve below, tilde, left angle above, bridge below
post_annotations = "ːˑ‿ʰʱʲʷˠˤ˞ⁿˡʼʴ\u0329\u032f\u0303\u031a\u032a"
ipa_symbols = vowels + consonants + pre_annotations + post_annotations

symbols = [separator] + list(_punctuation) + list(ipa_symbols)

symbol_to_id = {s: i for i, s in enumerate(symbols)}
voiced_phoneme_ids = {symbol_to_id[s] for s in vowels + consonants if s in symbol_to_id}
annotation_ids = {symbol_to_id[s] for s in pre_annotations + post_annotations}

SEPARATOR_ID = symbol_to_id[separator]
SPACE_ID = symbols.index(" ")

N_VOCAB = len(symbols)
