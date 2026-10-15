"""
Convert RoLEX lexicon entries (https://github.com/adrianastan/rolex) to eSpeak-ng Romanian
dictionary entries (phoneme names from espeak-ng 1.51 phsource/ph_romanian).

RoLEX row (tab separated, 6 columns):
    word_form  lemma  MSD  syllabification  lexical_stress  phonetic_transcription
    copii      copil  Ncmp-n  co.pii  c'opii  k o p i j

The transcription is space separated SAMPA-like symbols. Stress is not in the transcription;
it is marked with ' before the stressed vowel letter in the lexical_stress column, so we map
it to the n-th syllable nucleus of the transcription via the syllabification column.

RoLEX does not encode some allophones that the stock eSpeak rules produce (tap r, palatalised
consonant before final -i, ŋ before g, ...). The converter reproduces those eSpeak conventions,
so a converted word only differs from the rules' output where RoLEX really disagrees
(stress position, vowel vs semivowel, different phoneme). That keeps the phoneme inventory and
style the model was trained on.

Used by build_ro_listx.py. Quick check from the repo root:
    python espeak/src/rolex_to_espeak.py co.pii "c'opii" "k o p i j"
"""

import re
import sys
import unicodedata

# Syllable nuclei in RoLEX transcriptions.
NUCLEI = {"a", "e", "i", "o", "u", "@", "1"}
# Non-syllabic vowels / glides. Letter-wise they are vowels, which matters for the r rules.
GLIDES = {"j", "w", "e_X", "o_X", "i_0"}
VOWELISH = NUCLEI | GLIDES

# RoLEX symbol -> eSpeak phoneme, for symbols that map one to one.
SIMPLE = {
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u", "@": "@", "1": "y",
    "j": "j", "w": "w",
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g",
    "f": "f", "v": "v", "s": "s", "z": "z", "S": "S", "Z": "Z", "h": "h",
    "m": "m", "n": "n", "l": "l", "r": "r",
    "ts": "ts", "tS": "tS", "gZ": "dZ", "ks": "ks", "gz": "gz",
}
SUPPORTED = set(SIMPLE) | {"e_X", "o_X", "i_0", "k_j", "g_j"}

# Falling diphthongs: nucleus + j / w (not followed by a vowel) -> eSpeak diphthong phoneme.
# Same choices as ro_rules: ai->aI, ăi->@I, âi->yI, ei->eI, ui->uI, ii->iI, oi->oj,
# au->aU, ău->@U, âu->yU, ou->oU, iu->iU, eu->eU.
DIPHTHONG_J = {"a": "aI", "@": "@I", "1": "yI", "e": "eI", "u": "uI", "i": "iI", "o": "oj"}
DIPHTHONG_W = {"a": "aU", "@": "@U", "1": "yU", "o": "oU", "i": "iU", "e": "eU"}

# ro_rules "@) X (i_  X;" : consonant before a word final -i is palatalised.
PALATALISED = {"b": "b;", "d": "d;", "f": "f;", "v": "v;", "z": "z;", "Z": "Z;", "l": "l;",
               "m": "m;", "n": "n;", "p": "p;", "t": "t;", "ts": "ts;", "S": "S;", "h": "C"}


class ConversionError(ValueError):
    pass


def to_cedilla(text: str) -> str:
    """eSpeak's Romanian files use ş ţ (cedilla); ro_rules maps ș ț (comma) to them on input."""
    return unicodedata.normalize("NFC", text).replace("ș", "ş").replace("ț", "ţ").replace(
        "Ș", "Ş").replace("Ț", "Ţ")


VOWEL_LETTERS = set("aeiouăâîy")


def split_syllables(syllables: str) -> list[str]:
    """Split the syllabification column, merging vowel-less pieces into a neighbour.

    RoLEX sometimes splits off consonants as their own "syllable" (e.g. "șf.er.tu.lui").
    """
    merged = []
    pending = ""
    for piece in syllables.split("."):
        if any(c in VOWEL_LETTERS for c in piece):
            merged.append(pending + piece)
            pending = ""
        else:
            pending += piece
    if pending:
        if not merged:
            raise ConversionError("no vowel letter")
        merged[-1] += pending
    return merged


def stressed_syllable(syllables: list[str], stress: str) -> int:
    """Index of the stressed syllable. Monosyllables carry no mark in RoLEX."""
    marks = stress.count("'")
    if marks > 1:
        raise ConversionError("more than one stress mark")
    if stress.replace("'", "") != "".join(syllables):
        raise ConversionError("stress and syllabification columns spell different words")
    if marks == 0:
        if len(syllables) != 1:
            raise ConversionError("polysyllable without stress")
        return 0
    letters_before = stress.index("'")
    for idx, syllable in enumerate(syllables):
        if letters_before < len(syllable):
            return idx
        letters_before -= len(syllable)
    raise ConversionError("stress mark at end of word")


def convert_row(form: str, lemma: str, syllables: str, stress: str, transcription: str) -> str:
    """convert() plus row-level sanity checks on the RoLEX data."""
    f, l = form.lower(), lemma.lower()
    if l != "=" and l.startswith("î") and not f.startswith("î"):
        raise ConversionError("form lost its initial î")  # e.g. 'mpărți' for 'împărți'
    if stress.lower().replace("'", "") != f:
        raise ConversionError("stress column spells a different word")
    # RoLEX drops the short final i after an affricate (zeci -> z e tS, merge -> m e r gZ i_0
    # but mergi often -> m e r gZ). eSpeak keeps it as I^ (tʃʲ). Follow eSpeak, otherwise
    # listed and unlisted words would end differently.
    ph = transcription.split()
    if re.search(r"[cg]i$", f) and ph and ph[-1] in ("tS", "gZ"):
        transcription += " i_0"
    return convert(syllables, stress, transcription)


def convert(syllables: str, stress: str, transcription: str) -> str:
    """Return an eSpeak phoneme string (e.g. "k'opiI") for one RoLEX row."""
    ph = transcription.split()
    unknown = [p for p in ph if p not in SUPPORTED]
    if unknown:
        raise ConversionError(f"unsupported symbols {unknown}")
    # Foreign words keep their double letters in RoLEX (camorra -> r r, bitter -> t t), but
    # Romanian does not geminate; eSpeak rules also reduce them. Keep n n (înnoi, înnebuni).
    ph = [p for k, p in enumerate(ph)
          if k == 0 or p != ph[k - 1] or p in VOWELISH or p == "n"]
    # A final j after a consonant is the same non-syllabic -i as i_0 (adjuncţi, amâni).
    if len(ph) >= 2 and ph[-1] == "j" and ph[-2] not in VOWELISH:
        ph[-1] = "i_0"
    syllable_list = split_syllables(syllables.lower())
    stressed = stressed_syllable(syllable_list, stress.lower())
    n_nuclei = sum(p in NUCLEI for p in ph)
    if n_nuclei != len(syllable_list):
        raise ConversionError("nuclei/syllable count mismatch")

    out = []          # eSpeak phonemes
    nucleus_idx = -1  # running count of nuclei consumed
    i = 0

    def emit(phoneme, is_nucleus=False):
        nonlocal nucleus_idx
        stress_mark = ""
        if is_nucleus:
            nucleus_idx += 1
            if nucleus_idx == stressed:
                stress_mark = "'"
            elif nucleus_idx > stressed:
                # Explicitly unstressed. Without it eSpeak adds secondary stress after the
                # primary for dictionary words (abandonˈatelˌe), which the rules never do.
                stress_mark = "%"
        out.append((stress_mark, phoneme))

    def nxt(k=1):
        return ph[i + k] if i + k < len(ph) else None

    while i < len(ph):
        p = ph[i]
        prev = ph[i - 1] if i > 0 else None
        is_final_i = nxt() in ("i", "i_0") and i + 2 == len(ph)
        has_vowel_before = any(q in NUCLEI for q in ph[:i])

        if p == "e_X":
            if nxt() == "a":
                emit("ea", True); i += 2; continue
            if nxt() == "o":
                emit("eo", True); i += 2; continue
            raise ConversionError(f"e_X before {nxt()}")
        if p == "o_X":
            if nxt() == "a":
                emit("Oa", True); i += 2; continue
            raise ConversionError(f"o_X before {nxt()}")
        if p == "i_0":
            if i != len(ph) - 1:
                raise ConversionError("non-final i_0")
            emit("I^"); i += 1; continue
        if p in ("k_j", "g_j"):
            base = p[0]
            if nxt() in ("i", "e", "j", "i_0"):
                emit(base)
            elif nxt() in VOWELISH:
                emit(base); emit("j")
            elif nxt() is None:
                emit(base); emit("I^")        # unchi, ochi, unghi
            else:
                raise ConversionError(f"{p} before consonant")
            i += 1; continue
        if p in NUCLEI:
            follower = nxt()
            after = nxt(2)
            if follower == "j" and (after is None or after not in VOWELISH) and p in DIPHTHONG_J:
                d = DIPHTHONG_J[p]
                if d == "oj":
                    emit("o", True); emit("j")
                else:
                    emit(d, True)
                i += 2; continue
            if follower == "w" and (after is None or after not in VOWELISH) and p in DIPHTHONG_W:
                emit(DIPHTHONG_W[p], True); i += 2; continue
            emit(SIMPLE[p], True); i += 1; continue
        if p == "r":
            vowel_next = nxt() in VOWELISH
            vowel_prev = prev in VOWELISH
            if is_final_i and vowel_prev:
                emit("*;")                     # A) r (i_  *;
            elif vowel_next and vowel_prev:
                emit("*")                      # A) r (A   *
            elif vowel_next and prev is not None:
                emit("@-*")                    # C) r (A   @-*
            else:
                emit("r")
            i += 1; continue
        if p == "n" and nxt() in ("g", "g_j"):
            emit("N"); i += 1; continue        # n (g  N
        if is_final_i and has_vowel_before and p in PALATALISED:
            emit(PALATALISED[p]); i += 1; continue
        emit(SIMPLE[p]); i += 1

    return join_phonemes(out)


# eSpeak parses phoneme strings greedily, so "e"+"a" would become the diphthong "ea",
# "@"+"I^" the diphthong "@I" followed by a stray "^", "t"+"S" the affricate "tS".
# A "|" between two phonemes keeps them apart and is otherwise ignored.
VOWEL_CHARS = set("aeiou@yIUO^")
MERGING_CONSONANTS = {("t", "S"), ("t", "s"), ("t", "s;"), ("t", "S;"), ("d", "Z"), ("d", "Z;"),
                      ("k", "s"), ("g", "z")}


def join_phonemes(units: list[tuple[str, str]]) -> str:
    text = ""
    prev = None
    for stress_mark, phoneme in units:
        if prev is not None and (
                (prev[-1] in VOWEL_CHARS and phoneme[0] in VOWEL_CHARS)
                or (prev, phoneme) in MERGING_CONSONANTS):
            text += "|"
        text += stress_mark + phoneme
        prev = phoneme
    return text


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(f"usage: {sys.argv[0]} <syllabification> <stress> <transcription>")
    print(convert(*sys.argv[1:]))
