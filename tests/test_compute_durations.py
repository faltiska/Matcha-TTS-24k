"""
Tests for matcha.utils.compute_durations.

Run with:
  pytest tests/test_compute_durations.py
"""
import pytest
from unittest.mock import MagicMock
from matcha.utils.compute_durations import build_cer_strip, cer, _CER_STRIP
import matcha.utils.compute_durations as cd


def _make_processor(vocab_tokens):
    """Build a minimal mock processor whose tokenizer returns the given vocab."""
    proc = MagicMock()
    proc.tokenizer.get_vocab.return_value = {t: i for i, t in enumerate(vocab_tokens)}
    return proc


class TestW2V2NasalVowelEncoding:
    """Exploratory test: inspect how the real w2v2 tokenizer stores nasal vowels.
    Run with: pytest tests/test_compute_durations.py::TestW2V2NasalVowelEncoding -v -s
    Requires the model to be downloaded (~1GB).
    """

    @pytest.fixture(scope="class")
    def vocab(self):
        import os
        from pathlib import Path
        cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
        os.environ["HF_HOME"] = str(cache_base / "huggingface")
        from transformers import Wav2Vec2Processor
        proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
        return proc.tokenizer.get_vocab()

    def test_nasal_vowel_codepoints_in_vocab(self, vocab):
        """Print codepoint breakdown of all multi-codepoint vocab tokens (e.g. nasal vowels)."""
        import unicodedata
        multi = {t: v for t, v in vocab.items() if len(t) > 1}
        print(f"\n{len(multi)} multi-codepoint tokens in w2v2 vocab:")
        for token in sorted(multi):
            codepoints = " + ".join(f"U+{ord(c):04X} ({unicodedata.name(c, '?')})" for c in token)
            print(f"  {token!r} -> {codepoints}")
        # Not a hard assertion — just ensure there are some (nasal vowels like ɑ̃ ɛ̃ ɔ̃)
        assert len(multi) > 0


class TestBuildCerStrip:
    def test_multi_codepoint_nasal_vowels_preserved(self):
        """Combining tilde must not be stripped when it forms part of a known vocab token.

        French nasal vowels ɑ̃ ɛ̃ ɔ̃ œ̃ are each two codepoints (base + U+0303).
        Without the 'protected' guard, the combining tilde would be added to strip_chars
        (it is in matcha_symbols as a standalone diacritic but not as a solo vocab token),
        silently destroying all nasal vowels in the stripped string.
        """
        # Vocab contains the full multi-codepoint tokens, not the combining mark alone
        nasal_vowels = ["ɑ̃", "ɛ̃", "ɔ̃", "œ̃"]
        proc = _make_processor(nasal_vowels + ["a", "e", "o"])
        build_cer_strip(proc)

        # A string with all four nasal vowels plus stress/space (which should be stripped)
        phonemes = "ˈɑ̃ ɛ̃ ɔ̃ œ̃"
        stripped = phonemes.translate(cd._CER_STRIP).replace(" ", "")
        assert stripped == "ɑ̃ɛ̃ɔ̃œ̃", f"Nasal vowels were corrupted: {stripped!r}"

    def test_stress_markers_and_spaces_are_stripped(self):
        """Stress markers ˈ ˌ and spaces should still be stripped."""
        proc = _make_processor(["a", "e", "i"])
        build_cer_strip(proc)

        phonemes = "ˈa ˌe i"
        stripped = phonemes.translate(cd._CER_STRIP).replace(" ", "")
        assert stripped == "aei"
