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
- TestPhonemeIds: id-list integrity
- TestInsertSeparators: the separator rules, one case per situation, on real eSpeak output written by hand
- TestSeparatorRulesOnEspeakOutput: the separator rules checked on live eSpeak output for every supported language

Run with:
  pytest tests/test_phonemizers.py
"""
import pytest
from matcha.text.phonemizers import (
    cleanup_text,
    insert_separators,
    normalize_text,
    multilingual_phonemizer,
    phonemizers,
)
from matcha.text.symbols import (
    N_VOCAB,
    SEPARATOR_ID,
    annotation_ids,
    symbol_to_id,
    symbols,
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

    Verifies output shape and per-language operation rather than exact IPA snapshots,
    which drift every time NeMo or eSpeak releases an update with new pronunciations.
    """

    # --- Output shape ---

    def test_returns_string_and_list(self):
        phonemes, ids = multilingual_phonemizer("Hello world.", "en-us")
        assert isinstance(phonemes, str)
        assert isinstance(ids, list)

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError):
            multilingual_phonemizer("Hello.", "xx-xx")

    # --- Per-language smoke tests ---
    # Verify each supported language produces non-empty phoneme content. Exact IPA strings
    # are intentionally not asserted here — see the class docstring.

    def _assert_phonemizer_works_for_language(self, text, lang):
        phonemes, ids = multilingual_phonemizer(text, lang)
        content_without_padding = phonemes.strip(" ")
        assert len(content_without_padding) > 0, f"Empty phonemes for {lang}: {text!r}"
        assert len(ids) > 0, f"No phoneme IDs were produced for {lang}: {text!r}"
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

class TestNormalizeTextLanguageSupport:
    """Tests normalize_text behavior for languages with and without NeMo support."""

    def test_ro_text_returned_unchanged(self):
        # ro has no Nemo normalizer, text should pass through (minus left single quote removal)
        assert normalize_text("ro", "Temperatura este -5°C sau 23°F.") == "Temperatura este -5°C sau 23°F."

    def test_pt_text_is_normalized(self):
        assert normalize_text("pt", "Ele tem 5 gatos.") == "Ele tem cinco gatos."

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

    def test_displayed_phonemes_match_ids(self):
        phonemes, ids = multilingual_phonemizer("Hello world.", "en-us")
        assert phonemes == "".join(symbols[token_id] for token_id in ids)


def with_separators(espeak_output):
    """Runs the separator rules on an eSpeak output string and returns the result as a string."""
    espeak_ids = [symbol_to_id[symbol] for symbol in espeak_output]
    return "".join(symbols[token_id] for token_id in insert_separators(espeak_ids))


class TestInsertSeparators:
    """
    One case per situation. Inputs are eSpeak outputs, with the leading space the phonemizer adds. A few are short
    fragments built around annotation contexts found in the training corpus (stress between two vowels, Italian
    length mark before stress, Romanian palatalization around secondary stress).
    In the expected strings, "|" is the separator and " " is the word gap.
    """

    def test_between_two_phonemes(self):
        assert with_separators(" nɛkst") == " |n|ɛ|k|s|t"

    def test_around_word_gap(self):
        assert with_separators(" deɪ wiː") == " |d|e|ɪ| |w|iː"

    def test_before_final_punctuation(self):
        assert with_separators(" fˈʌn.") == " |fˈʌ|n|."

    def test_after_comma(self):
        assert with_separators(" həlˈoʊ, wˈɜːld!") == " |h|ə|lˈo|ʊ|, |wˈɜːl|d|!"

    def test_not_between_punctuation_and_word_gap(self):
        assert with_separators(" dʒ, ˈeɪt") == " |d|ʒ|, ˈe|ɪ|t"

    def test_not_around_stress_inside_word(self):
        assert with_separators(" tjˈuːzdeɪ") == " |t|jˈuːz|d|e|ɪ"

    def test_not_around_secondary_stress(self):
        assert with_separators(" sˌʌm") == " |sˌʌ|m"

    def test_not_between_word_gap_and_stress(self):
        assert with_separators(" ˈʌðə") == " ˈʌ|ð|ə"

    def test_not_around_stress_between_two_vowels(self):
        assert with_separators(" ɐˈa") == " |ɐˈa"

    def test_not_around_length_mark_inside_word(self):
        assert with_separators(" nˈiːd") == " |nˈiːd"

    def test_not_between_length_mark_and_word_gap(self):
        assert with_separators(" wiː nˈiːd") == " |w|iː |nˈiːd"

    def test_not_between_length_mark_and_punctuation(self):
        assert with_separators(" juː?") == " |j|uː?"

    def test_italian_double_consonant_length_mark(self):
        assert with_separators(" nˈɔtːe.") == " |nˈɔ|tːe|."

    def test_italian_length_mark_before_stress_is_kept_as_a_run(self):
        assert with_separators(" fatːˈo") == " |f|a|tːˈo"

    def test_english_syllabic_consonant_mark(self):
        assert with_separators(" bˈʌʔn̩ ænd") == " |bˈʌ|ʔ|n̩ |æ|n|d"

    def test_french_nasal_vowel_mark(self):
        assert with_separators(" œ̃ bˈɔ̃ vˈɛ̃.") == " |œ̃ |bˈɔ̃ |vˈɛ̃."

    def test_french_liaison_hyphen_is_punctuation(self):
        assert with_separators(" le-z amˈi") == " |l|e|-|z| |a|mˈi"

    def test_romanian_doubled_palatalization_is_kept_as_a_run(self):
        assert with_separators(" lˈupʲʲ ʃi") == " |lˈu|pʲʲ |ʃ|i"

    def test_romanian_annotation_run_with_stress_is_kept(self):
        assert with_separators(" mʲˌʲa") == " |mʲˌʲa"

    def test_diphthong_halves_are_separated(self):
        assert with_separators(" tˈaɪmz") == " |tˈa|ɪ|m|z"

    def test_affricate_halves_are_separated(self):
        assert with_separators(" tʃˈɜːtʃ") == " |t|ʃˈɜːt|ʃ"

    def test_leading_space_alone(self):
        assert with_separators(" ") == " "

    def test_no_separator_before_first_symbol(self):
        assert with_separators("a") == "a"

    def test_espeak_symbols_are_kept_in_order(self):
        espeak_output = " ɪn sˌʌm fˈɔːm ɔːɹ ˈʌðə wiː nˈiːd fˈʌn."
        espeak_ids = [symbol_to_id[symbol] for symbol in espeak_output]
        ids_without_separators = [token_id for token_id in insert_separators(espeak_ids) if token_id != SEPARATOR_ID]
        assert ids_without_separators == espeak_ids


# Sentences chosen to produce stress marks, length marks, nasal vowels, syllabic consonants, palatalization,
# double consonants, diphthongs, affricates, liaison and every kind of punctuation that survives to eSpeak's output.
SENTENCES_BY_LANGUAGE = {
    "en-us": ["Hello, world! Didn't you?", "The button and the rhythm.", "Church and judge; eight times: no."],
    "en-gb": ["In some form or other we need fun.", "Next Tuesday we must vote."],
    "fr-fr": ["Un bon vin blanc.", "Les amis arrivent, n'est-ce pas?"],
    "ro": ["Lupi și pomi.", "Copii frumoși, mii de ani!"],
    "it": ["Una bella ragazza, notte.", "Grazie mille, fatto!"],
    "es": ["Hola mundo."],
    "pt": ["Olá mundo, pão."],
    "de": ["Hallo Welt, schön."],
}


class TestSeparatorRulesOnEspeakOutput:
    """
    Requires NeMo + eSpeak. Checks the three rules on live output instead of exact strings, which drift with every
    eSpeak release: eSpeak's symbols are kept in order, and a separator sits between two neighbouring symbols
    exactly when one of them is a voiced phoneme and neither is an annotation.
    """

    @pytest.mark.parametrize(
        "text, language",
        [(text, language) for language, texts in SENTENCES_BY_LANGUAGE.items() for text in texts],
    )
    def test_rules_hold(self, text, language):
        _, ids = multilingual_phonemizer(text, language)
        espeak_ids = [token_id for token_id in ids if token_id != SEPARATOR_ID]

        language_code = language.split("-")[0]
        espeak_input = cleanup_text(normalize_text(language_code, text))
        espeak_output = " " + phonemizers[language].phonemize([espeak_input])[0].rstrip()
        assert espeak_ids == [symbol_to_id[symbol] for symbol in espeak_output]

        is_separator = [token_id == SEPARATOR_ID for token_id in ids]
        assert not is_separator[0] and not is_separator[-1]
        for position in range(1, len(ids) - 1):
            if not is_separator[position]:
                continue
            left_id, right_id = ids[position - 1], ids[position + 1]
            assert left_id != SEPARATOR_ID and right_id != SEPARATOR_ID, f"Two separators in a row in {text!r}"
            assert needs_separator(left_id, right_id), f"Unneeded separator between {symbols[left_id]!r} and {symbols[right_id]!r}"

        # Together with the checks above, an equal count means no needed separator is missing.
        expected_separator_count = sum(
            1 for left_id, right_id in zip(espeak_ids, espeak_ids[1:]) if needs_separator(left_id, right_id)
        )
        assert sum(is_separator) == expected_separator_count


def needs_separator(left_id, right_id):
    """The separator rule, written independently of the implementation under test."""
    touches_a_voiced_phoneme = left_id in voiced_phoneme_ids or right_id in voiced_phoneme_ids
    touches_an_annotation = left_id in annotation_ids or right_id in annotation_ids
    return touches_a_voiced_phoneme and not touches_an_annotation
