"""
Tests for split_ipa in matcha.text.phonemizers.

Run with:
  pytest tests/test_group_phonemes.py
"""
from matcha.text.phonemizers import group_phonemes


class TestPreAnnotations:
    def test_stress_marker_attaches_to_next_symbol(self):
        # "ˈa" must be one group, not "bˈ" + "a"
        assert group_phonemes("ˈaˌɪ") == ["ˈa", "ˌɪ"]

    def test_secondary_stress_attaches_to_next_symbol(self):
        assert group_phonemes("bˌat") == ["b", "ˌa", "t"]

    def test_stress_at_start_of_string(self):
        assert group_phonemes("ˈaʊt") == ["ˈa", "ʊ", "t"]

    def test_multiple_pre_annotations(self):
        assert group_phonemes("ˈˈaˌɪ") == ["ˈ", "ˈa", "ˌɪ"]

    def test_pre_after_post(self):
        assert group_phonemes("pʰˈɪn") == ["pʰ", "ˈɪ", "n"]

class TestPrePostAnnotations:
    def test_stress_and_length_mark(self):
        # "ˈɑː" is pre + base + post — one group
        assert group_phonemes("ˈɑːt") == ["ˈɑː", "t"]


class TestPostAnnotations:
    def test_aspiration_attaches_to_previous(self):
        assert group_phonemes("pʰɪn") == ["pʰ", "ɪ", "n"]

    def test_length_mark_attaches_to_previous(self):
        # German "Bahn" → "baːn"
        assert group_phonemes("baːaːn") == ["b", "aː", "aː", "n"]

    def test_palatalization_attaches_to_previous(self):
        assert group_phonemes("tʲa") == ["tʲ", "a"]

    def test_post_annotation_on_vowel(self):
        assert group_phonemes("ɑːt") == ["ɑː", "t"]

    def test_undertie_attaches_to_previous(self):
        # Undertie (U+203F) is backward-sticky via post_annotations, not via Unicode category
        assert group_phonemes("a‿b") == ["a‿", "b"]

    def test_multiple_post_annotations(self):
        assert group_phonemes("aɑːːɪ") == ["a", "ɑː", "ː", "ɪ"]

class TestCombiningCodepoints:
    def test_nasal_tilde_stays_with_base(self):
        # "ɑ̃" decomposes to "ɑ" + combining tilde under NFD
        assert group_phonemes("ɑ̃") == ["ɑ̃"]

    def test_french_nasal_vowel_in_context(self):
        # "un an" → "œ̃n ˈɑ̃"
        result = group_phonemes("œ̃n ˈɑ̃")
        assert result == ["œ̃", "n", " ", "ˈɑ̃"]

    def test_stress_on_combining_vowel(self):
        # Pre-annotation + combining codepoint must stay together
        assert group_phonemes("ˈɑ̃") == ["ˈɑ̃"]


class TestRomanianEspeakBugs:
    def test_double_palatalization_splits_into_two_groups(self):
        # eSpeak Romanian bug: emits "nʲʲ" instead of "nʲ"
        assert group_phonemes("nʲʲ") == ["nʲ", "ʲ"]

    def test_stray_pre_post_annotation_sequence_splits(self):
        # eSpeak Romanian bug: emits "ˌʲ" with no base phoneme between annotations
        assert group_phonemes("ˌʲ") == ["ˌ", "ʲ"]

    def test_stray_annotation_before_punctuation(self):
        # A stray post-annotation must not absorb the following punctuation
        assert group_phonemes("tʲ.") == ["tʲ", "."]

    def test_secondary_stress_palatalization_in_context(self):
        # "câțiva" as produced by eSpeak Romanian: "kɨtsˌʲva"
        assert group_phonemes("kɨtsˌʲva") == ["k", "ɨ", "t", "s", "ˌ", "ʲ", "v", "a"]


class TestEdgeCases:
    def test_empty_string(self):
        assert group_phonemes("") == []

    def test_space_passes_through(self):
        assert group_phonemes("a b") == ["a", " ", "b"]

    def test_punctuation_passes_through(self):
        assert group_phonemes("a.b") == ["a", ".", "b"]

    def test_pre_annotation_before_punctuation_does_not_fuse(self):
        # A stress marker with no following phoneme must not absorb the punctuation
        assert group_phonemes("aˈ.") == ["a", "ˈ", "."]

    def test_plain_phonemes_unchanged(self):
        assert group_phonemes("bat") == ["b", "a", "t"]

