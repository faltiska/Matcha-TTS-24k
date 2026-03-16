"""
Tests for split_ipa in matcha.text.phonemizers.

Run with:
  pytest tests/test_split_ipa.py
"""
from matcha.text.phonemizers import split_ipa


class TestPreAnnotations:
    def test_stress_marker_attaches_to_next_symbol(self):
        # "ˈa" must be one group, not "bˈ" + "a"
        assert split_ipa("bˈat") == ["b", "ˈa", "t"]

    def test_secondary_stress_attaches_to_next_symbol(self):
        assert split_ipa("bˌat") == ["b", "ˌa", "t"]

    def test_stress_at_start_of_string(self):
        assert split_ipa("ˈaʊt") == ["ˈa", "ʊ", "t"]

    def test_multiple_stress_markers(self):
        # "pronunciation" has both primary and secondary stress
        assert split_ipa("ˌbæd ˈɡʊd") == ["ˌb", "æ", "d", " ", "ˈɡ", "ʊ", "d"]


class TestPrePostAnnotations:
    def test_stress_and_length_mark(self):
        # "ˈɑː" is pre + base + post — one group
        assert split_ipa("ˈɑːt") == ["ˈɑː", "t"]

    def test_stress_and_aspiration(self):
        assert split_ipa("ˈpʰɪn") == ["ˈpʰ", "ɪ", "n"]


class TestPostAnnotations:
    def test_aspiration_attaches_to_previous(self):
        assert split_ipa("pʰɪn") == ["pʰ", "ɪ", "n"]

    def test_length_mark_attaches_to_previous(self):
        # German "Bahn" → "baːn"
        assert split_ipa("baːn") == ["b", "aː", "n"]

    def test_palatalization_attaches_to_previous(self):
        assert split_ipa("tʲa") == ["tʲ", "a"]

    def test_post_annotation_on_vowel(self):
        assert split_ipa("ɑːt") == ["ɑː", "t"]

    def test_undertie_attaches_to_previous(self):
        # Undertie (U+203F) is backward-sticky via post_annotations, not via Unicode category
        assert split_ipa("a‿b") == ["a‿", "b"]


class TestCombiningCodepoints:
    def test_nasal_tilde_stays_with_base(self):
        # "ɑ̃" decomposes to "ɑ" + combining tilde under NFD
        assert split_ipa("ɑ̃") == ["ɑ̃"]

    def test_french_nasal_vowel_in_context(self):
        # "un an" → "œ̃n ˈɑ̃"
        result = split_ipa("œ̃n ˈɑ̃")
        assert result == ["œ̃", "n", " ", "ˈɑ̃"]

    def test_stress_on_combining_vowel(self):
        # Pre-annotation + combining codepoint must stay together
        assert split_ipa("ˈɑ̃") == ["ˈɑ̃"]


class TestTieCharacters:
    def test_double_articulation_tie_stays_together(self):
        # Tie bar (U+0361) connects two consonants
        result = split_ipa("t͡ʃ")
        assert len(result) == 1
        assert result[0] == "t͡ʃ"

    def test_voiced_affricate_stays_together(self):
        # "judge" affricate
        result = split_ipa("d͡ʒ")
        assert len(result) == 1
        assert result[0] == "d͡ʒ"

    def test_double_vertical_line_is_not_a_tie(self):
        # U+2016 DOUBLE VERTICAL LINE contains "DOUBLE" in its Unicode name but is not a tie bar.
        # The character after it must start a new group, not be merged into the previous one.
        result = split_ipa("a‖b")
        assert result == ["a", "‖", "b"]

    def test_tie_followed_by_pre_annotation(self):
        # ˈ should start a new group, not get swallowed into the affricate
        result = split_ipa("t͡ˈʃ")
        assert result == ["t͡", "ˈʃ"], f"Got: {result}"
    
class TestRomanianEspeakBugs:
    def test_double_palatalization_splits_into_two_groups(self):
        # eSpeak Romanian bug: emits "nʲʲ" instead of "nʲ"
        assert split_ipa("nʲʲ") == ["nʲ", "ʲ"]

    def test_stray_pre_post_annotation_sequence_splits(self):
        # eSpeak Romanian bug: emits "ˌʲ" with no base phoneme between annotations
        assert split_ipa("ˌʲ") == ["ˌ", "ʲ"]

    def test_stray_annotation_before_punctuation(self):
        # A stray post-annotation must not absorb the following punctuation
        assert split_ipa("tʲ.") == ["tʲ", "."]

    def test_secondary_stress_palatalization_in_context(self):
        # "câțiva" as produced by eSpeak Romanian: "kɨtsˌʲva"
        assert split_ipa("kɨtsˌʲva") == ["k", "ɨ", "t", "s", "ˌ", "ʲ", "v", "a"]


class TestEdgeCases:
    def test_empty_string(self):
        assert split_ipa("") == []

    def test_space_passes_through(self):
        assert split_ipa("a b") == ["a", " ", "b"]

    def test_punctuation_passes_through(self):
        assert split_ipa("a.b") == ["a", ".", "b"]

    def test_pre_annotation_before_punctuation_does_not_fuse(self):
        # A stress marker with no following phoneme must not absorb the punctuation
        assert split_ipa("aˈ.") == ["a", "ˈ", "."]

    def test_plain_phonemes_unchanged(self):
        assert split_ipa("bat") == ["b", "a", "t"]

