"""
Defines the set of symbols used in text input to the model.
"""

# Padding token used when batching sequences of different lengths
# Shorter sequences are padded with _ to match the longest sequence in the batch
# Example: ["hɛˈloʊ", "haɪ"] → ["hɛˈloʊ", "haɪ___"] (padded to length 5)
_pad = "_"

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
ipa_symbols = (
    # Vowels
    "aeiouɑɐɒæəɘɚɛɜɝɞɨɪɔøɵɤʉʊyɶœɯʏʌᵻ"
    # Consonants
    "bβcçdðfɡɢɣhɦɧħɥjɟʝkʎlɭʟɬɫɮmɱnɳɲŋɴpɸqrɹɺɾɽɻʀʁsʂʃtʈθvʋⱱwʍxχzʐʒʑʔʕʢʡʙɕɖʜɰ"
    # Suprasegmentals
    "ˈˌːˑ‿"
    # Tone and stress markers
    "↓↑→↗↘˥˦˧˨˩"
    # Diacritics (combining and modifier)
    "ʰʱʲʷˠˤ˞ⁿˡʼʴ̩̯̃̚"
)

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(ipa_symbols)

# Special symbol ids
SPACE_ID = symbols.index(" ")
