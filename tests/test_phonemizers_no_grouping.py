"""
Tests for matcha.text.phonemizers.

Three test classes:
- TestCleanupText: pure unit tests, no external dependencies
- TestNormalizeText: requires NeMo
- TestMultilingualPhonemizer: requires NeMo + eSpeak (integration tests)

Run with:
  pytest tests/test_phonemizers_no_grouping.py
"""
import pytest
from matcha.text.phonemizers import cleanup_text, normalize_text, multilingual_phonemizer


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
    """Integration tests: require NeMo + eSpeak."""

    def _p(self, text, lang):
        phonemes, _ = multilingual_phonemizer(text, lang)
        return phonemes

    def test_en_plain(self):
        assert self._p("I live for live broadcasts.", "en-us") == " |a|ɪ| |l|ˈ|ɪ|v| |f|ɔ|ː|ɹ| |l|ˈ|a|ɪ|v| |b|ɹ|ˈ|ɔ|ː|d|k|æ|s|t|s|."

    def test_en_doctor(self):
        assert self._p("Dr. Jones will see you at 15:00.", "en-us") == " |d|ˈ|ɑ|ː|k|t|ɚ| |d|ʒ|ˈ|o|ʊ|n|z| |w|ɪ|l| |s|ˈ|i|ː| |j|u|ː| |æ|t| |f|ˈ|ɪ|f|t|i|ː|n| |ə|k|l|ˈ|ɑ|ː|k|."

    def test_en_price(self):
        assert self._p("The price is $5.00 as of Jan 21st, 2026.", "en-us") == " |ð|ə| |p|ɹ|ˈ|a|ɪ|s| |ɪ|z| |f|ˈ|a|ɪ|v| |d|ˈ|ɑ|ː|l|ɚ|z| |æ|z| |ʌ|v| |d|ʒ|ˈ|æ|n|j|u|ː|ˌ|ɛ|ɹ|i| |t|w|ˈ|ɛ|n|t|i| |f|ˈ|ɜ|ː|s|t|,| |t|w|ˈ|ɛ|n|t|i| |t|w|ˈ|ɛ|n|t|i| |s|ˈ|ɪ|k|s|."

    def test_en_temperature(self):
        assert self._p("The temperature is -5°C or 23°F.", "en-us") == " |ð|ə| |t|ˈ|ɛ|m|p|ɹ|ɪ|t|ʃ|ɚ|ɹ| |ɪ|z| |m|ˈ|a|ɪ|n|ə|s| |f|ˈ|a|ɪ|v| |d|ᵻ|ɡ|ɹ|ˈ|i|ː|z| |s|ˈ|ɛ|l|s|ɪ|ə|s| |ɔ|ː|ɹ| |t|w|ˈ|ɛ|n|t|i| |θ|ɹ|ˈ|i|ː| |d|ᵻ|ɡ|ɹ|ˈ|i|ː|z| |f|ˈ|æ|ɹ|ə|n|h|ˌ|a|ɪ|t|."

    def test_en_ellipsis(self):
        assert self._p("He thought… and then spoke.", "en-us") == " |h|i|ː| |θ|ˈ|ɔ|ː|t|,| |æ|n|d| |ð|ˈ|ɛ|n| |s|p|ˈ|o|ʊ|k|."

    def test_en_url(self):
        assert self._p("Visit http://example.com/path for details.", "en-us") == " |v|ˈ|ɪ|z|ɪ|t| |ˌ|e|ɪ|t|ʃ|t|ˌ|i|ː|t|ˌ|i|ː|p|ˈ|i|ː| |k|ˈ|o|ʊ|l|ə|n| |s|l|ˈ|æ|ʃ| |s|l|ˈ|æ|ʃ| |ɛ|ɡ|z|ˈ|æ|m|p|ə|l| |d|ˈ|ɑ|ː|t| |k|ˈ|ɑ|ː|m| |s|l|ˈ|æ|ʃ| |p|ˈ|æ|θ| |f|ɔ|ː|ɹ| |d|i|ː|t|ˈ|e|ɪ|l|z|."

    def test_en_backslash_path(self):
        assert self._p("C:\\Users\\name\\file.txt was found.", "en-us") == " |s|ˈ|i|ː|:|b|ˈ|æ|k|s|l|æ|ʃ| |j|ˈ|u|ː|z|ɚ|z| |b|ˈ|æ|k|s|l|æ|ʃ| |n|ˈ|e|ɪ|m| |b|ˈ|æ|k|s|l|æ|ʃ| |f|ˈ|a|ɪ|l|.|t|ˌ|i|ː|ˌ|ɛ|k|s|t|ˈ|i|ː| |w|ʌ|z| |f|ˈ|a|ʊ|n|d|."

    def test_en_brackets(self):
        assert self._p("The value is <10> or (20) or [30] or {40}.", "en-us") == " |ð|ə| |v|ˈ|æ|l|j|u|ː| |ɪ|z|,| |t|ˈ|ɛ|n|,| |ɔ|ː|ɹ|,| |t|w|ˈ|ɛ|n|t|i|,| |ɔ|ː|ɹ|,| |θ|ˈ|ɜ|ː|ɾ|i|,| |ɔ|ː|ɹ|,| |f|ˈ|ɔ|ː|ɹ|ɾ|i|."

    def test_en_em_dash(self):
        assert self._p("It was a dark and stormy night—except at occasional intervals.", "en-us") == " |ɪ|t| |w|ʌ|z|ɐ| |d|ˈ|ɑ|ː|ɹ|k| |æ|n|d| |s|t|ˈ|o|ː|ɹ|m|i| |n|ˈ|a|ɪ|t|,| |ɛ|k|s|ˈ|ɛ|p|t| |æ|ɾ| |ə|k|ˈ|e|ɪ|ʒ|ə|n|ə|l| |ˈ|ɪ|n|t|ɚ|v|ə|l|z|."

    def test_fr_guillemets_removed(self):
        assert self._p("Elle a dit «bonjour» à lui.", "fr-fr") == " |ɛ|l| |a| |d|ˈ|i| |b|ɔ|̃|ʒ|ˈ|u|ʁ| |a| |l|y|ˈ|i|."

    def test_fr_em_dash(self):
        assert self._p("La pluie tombait à torrents—sauf à intervalles occasionnels.", "fr-fr") == " |l|a|-| |p|l|y|ˈ|i| |t|ɔ|̃|b|ˈ|ɛ|t| |a| |t|o|ʁ|ˈ|ɑ|̃|,| |s|ˈ|o|f| |a| |ɛ|̃|t|ɛ|ʁ|v|ˈ|a|l|z| |ɔ|k|a|z|j|ɔ|n|ˈ|ɛ|l|."

    def test_ro_question(self):
        assert self._p("Oare?", "ro") == " |ˈ|ɔ|a|ɾ|e|?"

    def test_ro_exclamation(self):
        assert self._p("Doare!", "ro") == " |d|ˈ|ɔ|a|ɾ|e|!"

    def test_ro_hyphen(self):
        assert self._p("N-are.", "ro") == " |n|ˈ|a|ɾ|e|."

    def test_ro_trailing_whitespace(self):
        assert self._p("Cuvânt   ", "ro") == " |k|u|v|ˈ|ɨ|n|t|."

    def test_ro_em_dash(self):
        assert self._p("Ploaia cădea în torente—cu excepția momentelor ocazionale.", "ro") == " |p|l|ˈ|ɔ|a|j|a| |k|ə|d|ˈ|e|a| |ɨ|n| |t|o|ɾ|ˈ|e|n|t|e|,| |k|u| |e|k|s|t|ʃ|ˈ|e|p|t|s|j|a| |m|ˌ|o|m|e|n|t|ˈ|e|l|o|r| |ˌ|o|k|a|z|j|o|n|ˈ|a|l|e|."

    # --- missing EN cases from original main() ---

    def test_en_percent(self):
        assert self._p("He scored 95% on the test.", "en-us") == " |h|i|ː| |s|k|ˈ|o|ː|ɹ|d| |n|ˈ|a|ɪ|n|t|i| |f|ˈ|a|ɪ|v| |p|ɚ|s|ˈ|ɛ|n|t| |ɔ|n|ð|ə| |t|ˈ|ɛ|s|t|."

    def test_en_phone_address(self):
        assert self._p("Call me at 555-1234 or visit 123 Main St.", "en-us") == " |k|ˈ|ɔ|ː|l| |m|ˌ|i|ː| |æ|t| |f|ˈ|a|ɪ|v| |h|ˈ|ʌ|n|d|ɹ|ɪ|d| |æ|n|d| |f|ˈ|ɪ|f|t|i| |f|ˈ|a|ɪ|v| |t|w|ˈ|ɛ|l|v| |θ|ˈ|ɜ|ː|ɾ|i| |f|ˈ|o|ː|ɹ| |ɔ|ː|ɹ| |v|ˈ|ɪ|z|ɪ|t| |w|ˈ|ʌ|n| |t|w|ˈ|ɛ|n|t|i| |θ|ɹ|ˈ|i|ː| |m|ˈ|e|ɪ|n| |s|t|ɹ|ˈ|i|ː|t|."

    def test_en_years_en_dash(self):
        assert self._p("The years 2020—2025 were challenging.", "en-us") == " |ð|ə| |j|ˈ|ɪ|ɹ|z| |t|w|ˈ|ɛ|n|t|i| |t|w|ˈ|ɛ|n|t|i|,| |t|w|ˈ|ɛ|n|t|i| |t|w|ˈ|ɛ|n|t|i| |f|ˈ|a|ɪ|v| |w|ɜ|ː| |t|ʃ|ˈ|æ|l|ə|n|d|ʒ|ˌ|ɪ|ŋ|."

    def test_en_smart_quotes_and_right_single(self):
        assert self._p("He said “hello” to me and I've said ‘hello’ back.", "en-us") == " |h|i|ː| |s|ˈ|ɛ|d| |h|ə|l|ˈ|o|ʊ| |t|ə| |m|ˌ|i|ː| |æ|n|d| |a|ɪ|v| |s|ˈ|ɛ|d| |h|ə|l|ˈ|o|ʊ| |b|ˈ|æ|k|."

    # --- RO missing whitespace variants ---

    def test_ro_trailing_newlines(self):
        assert self._p("Cuvânt\n\n", "ro") == " |k|u|v|ˈ|ɨ|n|t|."

    def test_ro_trailing_tab(self):
        assert self._p("Cuvânt\t", "ro") == " |k|u|v|ˈ|ɨ|n|t|."

    # --- ES phonemizer ---

    def test_es_doctor(self):
        assert self._p("El Dr. García llegará a las 15:00.", "es") == " |e|l| |ð|o|k|t|ˈ|o|ɾ| |ɣ|a|ɾ|θ|ˈ|i|a| |ʎ|ˌ|e|ɣ|a|ɾ|ˈ|a| |a| |l|a|s| |k|ˈ|i|n|θ|e|."

    def test_es_precio(self):
        # NeMo ES does not handle $5.00 well - documents current behavior
        assert self._p("El precio es $5.00 desde el 21 de enero de 2026.", "es") == " |e|l| |p|ɾ|ˈ|e|θ|j|o| |ˈ|e|s| |s|ˈ|i|ɡ|n|o| |ð|e| |ð|ˈ|o|l|a|ɾ| |θ|ˈ|i|n|k|o| |p|ˈ|u|n|t|o| |θ|ˈ|e|ɾ|o| |θ|ˈ|e|ɾ|o| |ð|ˌ|e|s|ð|e| |e|l| |β|e|ɪ|n|t|j|ˈ|u|n|o| |ð|e| |e|n|ˈ|e|ɾ|o| |ð|e| |ð|ˈ|o|s| |m|ˈ|i|l| |β|ˌ|e|ɪ|n|t|i|s|ˈ|e|i|s|."

    def test_es_temperatura(self):
        assert self._p("La temperatura es -5°C o 23°F.", "es") == " |l|a| |t|ˌ|e|m|p|e|ɾ|a|t|ˈ|u|ɾ|a| |ˈ|e|s| |θ|ˈ|i|n|k|o| |ɣ|ɾ|ˈ|a|ð|o|s| |θ|ˈ|e| |o| |β|ˌ|e|ɪ|n|t|i|t|ɾ|ˈ|e|s| |ɣ|ɾ|ˈ|a|ð|o|s| |ˈ|ɛ|f|e|."

    # --- DE phonemizer ---

    def test_de_doctor(self):
        assert self._p("Dr. M\u00fcller sieht Sie um 15:00 Uhr.", "de") == " |d|ˈ|ɔ|k|t|o|ː|ɾ| |m|ˈ|y|l|ɜ| |z|ˈ|i|ː|t| |z|i|ː| |ʊ|m| |f|ˈ|y|n|f|t|s|e|ː|n| |ˈ|u|ː|ɾ|."

    def test_de_preis(self):
        assert self._p("Der Preis beträgt 5,00€ ab dem 21. Januar 2026.", "de") == " |d|ɛ|ɾ| |p|ɾ|ˈ|a|ɪ|s| |b|ə|t|ɾ|ˈ|ɛ|ː|k|t| |f|ˈ|y|n|f|,| |n|ˈ|ʊ|l| |n|ˈ|ʊ|l| |ˈ|ɔ|ø|r|o|ː| |a|p| |d|e|ː|m| |a|ɪ|n| |ʊ|n|t| |t|s|v|ˈ|a|n|t|s|ɪ|ç|s|t|ɜ| |j|ˈ|a|n|u|ː|ˌ|ɑ|ː|ɾ| |t|s|v|ˈ|a|ɪ| |t|ˈ|a|ʊ|z|ə|n|t| |z|ˈ|ɛ|k|s| |ʊ|n|t| |t|s|v|ˈ|a|n|t|s|ɪ|ç|s|t|ə|."

    def test_de_temperatur(self):
        assert self._p("Die Temperatur beträgt -5°C oder 23°F.", "de") == " |d|i|ː| |t|ˌ|ɛ|m|p|e|ː|r|a|t|ˈ|u|ː|ɾ| |b|ə|t|ɾ|ˈ|ɛ|ː|k|t| |m|ˈ|i|ː|n|ʊ|s| |f|ˈ|y|n|f| |ɡ|ɾ|ˈ|ɑ|ː|t| |t|s|ˈ|ɛ|l|z|i|ː|ˌ|ʊ|s| |ˌ|o|ː|d|ɜ| |d|ɾ|ˈ|a|ɪ| |ʊ|n|t| |t|s|v|ˈ|a|n|t|s|ɪ|ç| |ɡ|ɾ|ˈ|ɑ|ː|t| |f|ˈ|ɑ|ː|r|ə|n|h|ˌ|a|ɪ|t|."

    # --- IT phonemizer ---

    def test_it_doctor(self):
        assert self._p("Il Dr. Rossi la vedrà alle 15:00.", "it") == " |i|l| |d|o|t|ː|ˈ|ɔ|r| |r|ˈ|o|s|s|ɪ| |l|a| |v|e|d|r|ˈ|a| |ˌ|a|l|l|e| |k|w|ˈ|i|n|d|i|t|ʃ|ɪ|."

    def test_it_price(self):
        assert self._p("Il prezzo è €5,00 dal 21 gennaio 2026.", "it") == " |i|l| |p|r|ˈ|ɛ|t|s|ː|o| |e| |t|ʃ|ˈ|i|n|k|w|e| |ˈ|ɛ|ʊ|r|o| |d|z|ˈ|ɛ|r|o| |d|z|ˈ|ɛ|r|o| |d|a|l| |v|e|n|t|ˈ|u|n|o| |d|ʒ|e|n|n|ˈ|a|i|o| |d|ʊ|e|m|ˈ|i|l|a| |v|e|n|t|ɪ|s|ˈ|ɛ|j|."

    def test_it_temperature(self):
        assert self._p("La temperatura è -5°C o 23°F.", "it") == " |l|a| |t|e|m|p|e|r|a|t|ˈ|u|r|a| |e| |m|ˈ|e|n|o| |t|ʃ|ˈ|i|n|k|w|e| |ɡ|r|ˈ|a|d|o| |t|s|e|l|s|j|ˈ|u|s| |o| |v|e|n|t|i|t|r|ˈ|e| |ɡ|r|ˈ|a|d|o| |f|ˈ|a|r|e|n|a|ɪ|t|."

    # --- PT phonemizer (no Nemo, eSpeak only) ---

    def test_pt_doctor(self):
        assert self._p("O Dr. Silva verá você às 15:00.", "pt") == " |ʊ| |d|o|w|t|ˈ|o|r|.| |s|ˈ|i|l|v|ɐ| |v|ɨ|ɾ|ˈ|a| |v|o|s|ˌ|e| |ɐ|ɐ|ʃ| |k|ˈ|i|ŋ|z|ɨ|:|z|ˈ|ɛ|ɾ|u| |z|ˈ|ɛ|ɾ|u|."

    def test_pt_price(self):
        assert self._p("O preço é R$ 5,00 desde 21 de janeiro de 2026.", "pt") == " |ʊ| |p|ɹ|ˈ|e|s|w| |ɛ| |ʁ|ɨ|ˈ|a|ʊ| |s|ˈ|i|ŋ|k|u|,| |z|ˈ|ɛ|ɾ|u| |z|ˈ|ɛ|ɾ|u| |d|ˈ|e|ʒ|d|ɨ| |v|ˈ|i|ŋ|t|ɨ|i|ˈ|u|m| |d|ɨ| |ʒ|ɐ|n|ˈ|e|ɪ|ɾ|ʊ| |d|ɨ| |d|ˈ|o|ɪ|ʒ| |m|ˈ|i|l| |i| |v|ˈ|i|ŋ|t|ɨ|i|s|ˈ|e|ɪ|ʃ|."

    def test_pt_temperature(self):
        assert self._p("A temperatura é -5°C ou 23°F.", "pt") == " |ɐ| |t|ˌ|e|ɪ|m|p|ɨ|ɾ|ɐ|t|ˈ|u|ɾ|ɐ| |ɛ| |m|ˈ|e|n|ʊ|s| |s|ˈ|i|ŋ|k|u| |ɡ|ɹ|ˈ|a|ʊ| |s|ˈ|e| |ˈ|o|w| |v|ˈ|i|ŋ|t|ɨ|i|t|ɹ|ˈ|e|ʒ| |ɡ|ɹ|ˈ|a|ʊ| |ˈ|ɛ|f|."

    def test_en_syllabic_n(self):
        # This specific sentence triggers a syllabic n (ˌn̩) in eSpeak output.
        # validate_group must keep ˌn̩ as a single token since it is in symbols.
        result = self._p("He unfolded a long typewritten letter, and handed it to Gregson.", "en-us")
        tokens = result.split("|")
        assert "ˌ" in tokens
        assert "n" in tokens
        assert "\u0329" in tokens        

    def test_fr_semivowel_stress(self):
        # This specific sentence triggers a stressed semivowel (ˈw) in eSpeak output.
        # validate_group must keep ˈw as a single token since it is in symbols.
        result = self._p("Le sens de l'ouïe est plus fin que celui du toucher.", "fr-fr")
        tokens = result.split("|")
        assert "ˈ" in tokens
        assert "w" in tokens
        
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


class TestMultilingualPhonemizerErrors:
    """Tests for error handling in multilingual_phonemizer."""

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError):
            multilingual_phonemizer("Hello.", "xx-xx")


class TestPhonemizerOutputSymbols:
    """Document which _punctuation chars survive cleanup_text + eSpeak to the final output."""

    def test_only_prosodic_punctuation_survives_to_output(self):
        from matcha.text.symbols import _punctuation
        result, _ = multilingual_phonemizer(_punctuation, "en-us")
        surviving = set(result) & set(_punctuation)
        assert surviving == set(' ;:,.!?'), f"Unexpected surviving punctuation: {surviving}"

class TestPhonemeIds:
    """Tests for the ID sequence returned by multilingual_phonemizer."""

    def test_no_none_ids(self):
        _, ids = multilingual_phonemizer("Oare?", "ro")
        assert all(id is not None for id in ids)

    def test_no_trailing_separator(self):
        separator_id = 0  # | is the first symbol
        _, ids = multilingual_phonemizer("Oare?", "ro")
        assert ids[-1] != separator_id

    def test_alternating_token_separator_pattern(self):
        separator_id = 0
        _, ids = multilingual_phonemizer("Oare?", "ro")
        for i, id in enumerate(ids):
            is_separator_position = i % 2 == 1
            if is_separator_position:
                assert id == separator_id
            else:
                assert id != separator_id

    def test_ids_consistent_with_phoneme_string(self):
        from matcha.text.symbols import symbol_to_id
        phonemes, ids = multilingual_phonemizer("Oare?", "ro")
        tokens = phonemes.split("|")
        expected_ids = [id for token in tokens for id in (symbol_to_id[token], 0)][:-1]
        assert ids == expected_ids


class TestValidateGroup:
    """Tests for the validate_group safety check in split_ipa."""

    def test_unknown_group_is_split(self):
        from matcha.text.phonemizers import group_phonemes
        from matcha.text.symbols import symbol_to_id
        # ˈb is not in symbols, because stress cannot precede a plain consonant, so we expect split_ipa to 
        # to split the group and keep individual phonemes.
        tokens = group_phonemes("ˈb")
        assert tokens == ["ˈ", "b"]
        assert symbol_to_id.get("ˈb") is None
