"""
Tests for matcha.text.phonemizers.

Test classes:
- TestCleanupText: pure unit tests, no external dependencies
- TestNormalizeText: requires NeMo
- TestNormalizeTextFallback: behavior when NeMo is not available for a language
- TestMultilingualPhonemizer: requires NeMo + eSpeak. Verifies the structural contract
  (silence padding, output shape, per-language smoke) rather than exact IPA snapshots,
  which drift every time NeMo or eSpeak releases an update.
- TestPhonemizerOutputSymbols: documents which punctuation survives to the model
- TestPhonemeIds: id-list integrity, including voiced-phoneme tuple expansion

Run with:
  pytest tests/test_phonemizers.py
"""
import pytest
from matcha.text.phonemizers import (
    cleanup_text,
    normalize_text,
    multilingual_phonemizer,
    LEADING_SILENCE_SPACES,
    TRAILING_SILENCE_SPACES,
)
from matcha.text.symbols import (
    SPACE_ID,
    PRE_ID,
    POST_ID,
    N_VOCAB,
    symbol_to_id,
    voiced_phoneme_ids,
)


class TestCleanupText:
    def test_plain_sentence_unchanged(self):
        assert cleanup_text("I live for live broadcasts.") == "I live for live broadcasts."

    def test_adds_period_when_missing(self):
        assert cleanup_text("Word") == "Word."

    def test_trailing_spaces(self):
        assert cleanup_text("Word   ") == "Word."

    def test_trailing_newlines(self):
        assert cleanup_text("Word\n\n") == "Word."

    def test_trailing_tab(self):
        assert cleanup_text("Word\t") == "Word."

    def test_preserves_question_mark(self):
        assert cleanup_text("Oare?") == "Oare?"

    def test_preserves_exclamation(self):
        assert cleanup_text("Doare!") == "Doare!"

    def test_en_left_single_quote_removed(self):
        assert cleanup_text("He said “hello” to me and I've said ‘hello’ back.") == "He said hello to me and I've said ‘hello’ back."

    def test_removes_fancy_quotes(self):
        assert cleanup_text('„“Hello”.') == "Hello."

    def test_removes_guillemets(self):
        assert cleanup_text("Elle a dit «bonjour» à lui.") == "Elle a dit bonjour à lui."

    def test_angle_brackets_become_comma(self):
        assert cleanup_text("The value is <10>.") == "The value is, 10."

    def test_parens_become_comma(self):
        assert cleanup_text("The value is (10).") == "The value is, 10."

    def test_square_brackets_become_comma(self):
        assert cleanup_text("The value is [10].") == "The value is, 10."

    def test_curly_brackets_become_comma(self):
        assert cleanup_text("The value is {10}.") == "The value is, 10."

    def test_em_dash_becomes_comma(self):
        assert cleanup_text("night—except.") == "night, except."

    def test_en_dash_becomes_comma(self):
        assert cleanup_text("2020–2025.") == "2020, 2025."

    def test_ellipsis_becomes_comma(self):
        assert cleanup_text("He thought… and then spoke.") == "He thought, and then spoke."

    def test_leading_bracket_no_leading_comma(self):
        assert cleanup_text("<10> items.") == "10, items."

    def test_double_comma_collapsed(self):
        assert cleanup_text("a,, b.") == "a, b."

    def test_comma_before_period_removed(self):
        assert cleanup_text("end,.") == "end."

    def test_comma_before_question_removed(self):
        assert cleanup_text("end,?") == "end?"

    def test_hyphen_preserved(self):
        assert cleanup_text("N-are.") == "N-are."

    def test_space_before_period_removed(self):
        assert cleanup_text("Hello .") == "Hello."
    
    def test_space_before_question_removed(self):
        assert cleanup_text("Oare ?") == "Oare?"
    
    def test_space_before_exclamation_removed(self):
        assert cleanup_text("Doare !") == "Doare!"
    
    def test_space_before_comma_removed(self):
        assert cleanup_text("Hello , world.") == "Hello, world."
    
    def test_space_before_colon_removed(self):
        assert cleanup_text("Hello : world.") == "Hello: world."
    
    def test_space_before_semicolon_removed(self):
        assert cleanup_text("Hello ; world.") == "Hello; world."
    
    def test_em_dash_with_spaces_becomes_comma(self):
        assert cleanup_text("night — except.") == "night, except."
    
    def test_parens_with_spaces_become_comma(self):
        assert cleanup_text("The value is ( 10 ).") == "The value is, 10."
        
        
class TestNormalizeText:
    """Requires NeMo. Skip if not available."""

    def test_en_doctor_title(self):
        assert normalize_text("en", "Dr. Jones will see you at 15:00.") == "doctor Jones will see you at fifteen o'clock."

    def test_en_price(self):
        assert normalize_text("en", "The price is $5.00 as of Jan 21st, 2026.") == "The price is five dollars as of january twenty first, twenty twenty six."

    def test_en_phone_address(self):
        assert normalize_text("en", "Call me at 555-1234 or visit 123 Main St.") == "Call me at five hundred and fifty five - twelve thirty four or visit one twenty three Main Street"

    def test_en_temperature(self):
        assert normalize_text("en", "The temperature is -5°C or 23°F.") == "The temperature is minus five degrees Celsius or twenty three degrees Fahrenheit."

    def test_en_percent(self):
        assert normalize_text("en", "He scored 95% on the test.") == "He scored ninety five percent on the test."

    def test_en_trailing_whitespace_stripped(self):
        assert normalize_text("en", "Word   ") == "Word"

    def test_en_left_single_quote_removed(self):
        # Nemo mishandles ' (left smart quote) as if it were a straight apostrophe,
        # so we remove it before passing to Nemo. The right one ' is kept (used as apostrophe).
        assert normalize_text("en", "He said hello ‘back’.") == "He said hello back’."

    def test_en_url(self):
        assert normalize_text("en", "Visit http://example.com/path for details.") == "Visit HTTP colon slash slash example dot com slash PATH for details."

    def test_de_doctor_title(self):
        assert normalize_text("de", "Dr. Müller sieht Sie um 15:00 Uhr.") == "doktor Müller sieht Sie um fünfzehn uhr ."

    def test_de_temperature(self):
        assert normalize_text("de", "Die Temperatur beträgt -5°C oder 23°F.") == "Die Temperatur beträgt minus fünf grad celsius oder drei und zwanzig grad fahrenheit ."

    def test_fr_no_normalization_for_doctor(self):
        # NeMo fr does not expand Dr.
        assert normalize_text("fr", "Le Dr. Dupont vous verra à 15h00.") == "Le Dr. Dupont vous verra à 15h00."

    def test_it_doctor_title(self):
        assert normalize_text("it", "Il Dr. Rossi la vedrà alle 15:00.") == "Il dottor Rossi la vedrà alle quindici ."

    def test_es_doctor_title(self):
        assert normalize_text("es", "El Dr. García llegará a las 15:00.") == "El Doctor García llegará a las quince ."


class TestMultilingualPhonemizer:
    """Integration tests for multilingual_phonemizer: require NeMo + eSpeak.

    Verifies the structural contract — silence padding, output shape, per-language
    operation — rather than exact IPA snapshots, which would drift every time NeMo
    or eSpeak releases an update with new pronunciations.
    """

    # --- Output shape ---

    def test_returns_string_and_list(self):
        phonemes, ids = multilingual_phonemizer("Hello world.", "en-us")
        assert isinstance(phonemes, str)
        assert isinstance(ids, list)

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError):
            multilingual_phonemizer("Hello.", "xx-xx")

    # --- Silence padding contract ---

    def test_phonemes_string_has_correct_leading_space_count(self):
        phonemes, _ = multilingual_phonemizer("Hello world.", "en-us")
        leading_space_count = len(phonemes) - len(phonemes.lstrip(" "))
        assert leading_space_count == LEADING_SILENCE_SPACES

    def test_phonemes_string_has_correct_trailing_space_count(self):
        phonemes, _ = multilingual_phonemizer("Hello world.", "en-us")
        trailing_space_count = len(phonemes) - len(phonemes.rstrip(" "))
        assert trailing_space_count == TRAILING_SILENCE_SPACES

    def test_ids_start_with_leading_space_padding(self):
        _, ids = multilingual_phonemizer("Hello world.", "en-us")
        assert ids[:LEADING_SILENCE_SPACES] == [SPACE_ID] * LEADING_SILENCE_SPACES

    def test_ids_end_with_trailing_space_padding(self):
        _, ids = multilingual_phonemizer("Hello world.", "en-us")
        assert ids[-TRAILING_SILENCE_SPACES:] == [SPACE_ID] * TRAILING_SILENCE_SPACES

    def test_silence_padding_is_independent_of_input_length(self):
        # The padding is a fixed prefix/suffix; both very short and very long inputs
        # get exactly the same silence pad counts at each end.
        short_phonemes, _ = multilingual_phonemizer("I.", "en-us")
        long_phonemes, _ = multilingual_phonemizer(
            "This is a much longer sentence with many phonemes to process.", "en-us"
        )
        assert len(short_phonemes) - len(short_phonemes.lstrip(" ")) == LEADING_SILENCE_SPACES
        assert len(long_phonemes) - len(long_phonemes.lstrip(" ")) == LEADING_SILENCE_SPACES
        assert len(short_phonemes) - len(short_phonemes.rstrip(" ")) == TRAILING_SILENCE_SPACES
        assert len(long_phonemes) - len(long_phonemes.rstrip(" ")) == TRAILING_SILENCE_SPACES

    # --- Per-language smoke tests ---
    # Verify each supported language produces non-empty phoneme content. Exact IPA strings
    # are intentionally not asserted here — see the class docstring.

    def _assert_phonemizer_works_for_language(self, text, lang):
        phonemes, ids = multilingual_phonemizer(text, lang)
        content_without_padding = phonemes.strip(" ")
        assert len(content_without_padding) > 0, f"Empty phonemes for {lang}: {text!r}"
        assert len(ids) > LEADING_SILENCE_SPACES + TRAILING_SILENCE_SPACES, (
            f"Only padding IDs were produced for {lang}: {text!r}"
        )
        assert all(token_id is not None for token_id in ids), (
            f"None ID found in output for {lang}: {text!r}"
        )

    def test_en_us(self):
        self._assert_phonemizer_works_for_language("Hello world.", "en-us")

    def test_en_gb(self):
        self._assert_phonemizer_works_for_language("Hello world.", "en-gb")

    def test_fr_fr(self):
        self._assert_phonemizer_works_for_language("Bonjour le monde.", "fr-fr")

    def test_es(self):
        self._assert_phonemizer_works_for_language("Hola mundo.", "es")

    def test_pt(self):
        self._assert_phonemizer_works_for_language("Olá mundo.", "pt")

    def test_de(self):
        self._assert_phonemizer_works_for_language("Hallo Welt.", "de")

    def test_it(self):
        self._assert_phonemizer_works_for_language("Ciao mondo.", "it")

    def test_ro(self):
        self._assert_phonemizer_works_for_language("Salut lume.", "ro")

class TestNormalizeTextFallback:
    """Tests for normalize_text behavior when Nemo is not available for a language."""

    def test_ro_text_returned_unchanged(self):
        # ro has no Nemo normalizer, text should pass through (minus left single quote removal)
        assert normalize_text("ro", "Temperatura este -5°C sau 23°F.") == "Temperatura este -5°C sau 23°F."

    def test_pt_text_returned_unchanged(self):
        assert normalize_text("pt", "O Dr. Silva verá você às 15:00.") == "O Dr. Silva verá você às 15:00."

    def test_left_single_quote_removed_for_non_nemo_language(self):
        # The ' removal runs for ALL languages, not just EN
        assert normalize_text("ro", "N-‘are.") == "N-are."


class TestPhonemizerOutputSymbols:
    """Document which _punctuation chars survive cleanup_text + eSpeak to the final output."""

    def test_only_prosodic_punctuation_survives_to_output(self):
        from matcha.text.symbols import _punctuation
        result, _ = multilingual_phonemizer(_punctuation, "en-us")
        surviving = set(result) & set(_punctuation)
        assert surviving == set(' ;:,.!?'), f"Unexpected surviving punctuation: {surviving}"

class TestPhonemeIds:
    """Id-list integrity tests for multilingual_phonemizer."""

    def test_no_none_ids(self):
        _, ids = multilingual_phonemizer("Oare?", "ro")
        assert all(token_id is not None for token_id in ids)

    def test_ids_within_valid_range(self):
        _, ids = multilingual_phonemizer("Hello world.", "en-us")
        for token_id in ids:
            assert 0 <= token_id < N_VOCAB

    def test_voiced_phonemes_expand_to_three_ids(self):
        """
        Each voiced phoneme p in the displayed phonemes string is bracketed by '‹' '›' and
        contributes the triple (PRE_ID + p, p, POST_ID + p) to the id list.
        Each non-voiced symbol contributes a single id.
        We verify the contract by reconstructing the expected id list from the phonemes string.
        """
        phonemes, ids = multilingual_phonemizer("Hello world.", "en-us")

        expected_ids = []
        char_index = 0
        while char_index < len(phonemes):
            char = phonemes[char_index]
            if char == '‹':
                voiced_phoneme = phonemes[char_index + 1]
                voiced_phoneme_id = symbol_to_id[voiced_phoneme]
                assert voiced_phoneme_id in voiced_phoneme_ids
                expected_ids.extend([
                    PRE_ID + voiced_phoneme_id,
                    voiced_phoneme_id,
                    POST_ID + voiced_phoneme_id,
                ])
                # Skip past the closing '›' as well
                char_index += 3
            else:
                expected_ids.append(symbol_to_id[char])
                char_index += 1

        assert ids == expected_ids
