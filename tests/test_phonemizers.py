"""
Tests for matcha.text.phonemizers.

Three test classes:
- TestCleanupText: pure unit tests, no external dependencies
- TestNormalizeText: requires NeMo
- TestMultilingualPhonemizer: requires NeMo + eSpeak (integration tests)

Run with:
  pytest tests/test_phonemizers.py
  pytest tests/test_phonemizers.py -k TestCleanupText   # fast, no deps
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
        return multilingual_phonemizer(text, lang)

    def test_en_plain(self):
        assert self._p("I live for live broadcasts.", "en-us") == "a|ɪ| |l|ˈɪ|v| |f|ɔː|ɹ| |l|ˈa|ɪ|v| |b|ɹ|ˈɔː|d|k|æ|s|t|s|."

    def test_en_doctor(self):
        assert self._p("Dr. Jones will see you at 15:00.", "en-us") == "d|ˈɑː|k|t|ɚ| |d|ʒ|ˈo|ʊ|n|z| |w|ɪ|l| |s|ˈiː| |j|uː| |æ|t| |f|ˈɪ|f|t|iː|n| |ə|k|l|ˈɑː|k|."

    def test_en_price(self):
        assert self._p("The price is $5.00 as of Jan 21st, 2026.", "en-us") == "ð|ə| |p|ɹ|ˈa|ɪ|s| |ɪ|z| |f|ˈa|ɪ|v| |d|ˈɑː|l|ɚ|z| |æ|z| |ʌ|v| |d|ʒ|ˈæ|n|j|uː|ˌɛ|ɹ|i| |t|w|ˈɛ|n|t|i| |f|ˈɜː|s|t|,| |t|w|ˈɛ|n|t|i| |t|w|ˈɛ|n|t|i| |s|ˈɪ|k|s|."

    def test_en_temperature(self):
        assert self._p("The temperature is -5°C or 23°F.", "en-us") == "ð|ə| |t|ˈɛ|m|p|ɹ|ɪ|t|ʃ|ɚ|ɹ| |ɪ|z| |m|ˈa|ɪ|n|ə|s| |f|ˈa|ɪ|v| |d|ᵻ|ɡ|ɹ|ˈiː|z| |s|ˈɛ|l|s|ɪ|ə|s| |ɔː|ɹ| |t|w|ˈɛ|n|t|i| |θ|ɹ|ˈiː| |d|ᵻ|ɡ|ɹ|ˈiː|z| |f|ˈæ|ɹ|ə|n|h|ˌa|ɪ|t|."

    def test_en_ellipsis(self):
        assert self._p("He thought… and then spoke.", "en-us") == "h|iː| |θ|ˈɔː|t|,| |æ|n|d| |ð|ˈɛ|n| |s|p|ˈo|ʊ|k|."

    def test_en_url(self):
        assert self._p("Visit http://example.com/path for details.", "en-us") == "v|ˈɪ|z|ɪ|t| |ˌe|ɪ|t|ʃ|t|ˌiː|t|ˌiː|p|ˈiː| |k|ˈo|ʊ|l|ə|n| |s|l|ˈæ|ʃ| |s|l|ˈæ|ʃ| |ɛ|ɡ|z|ˈæ|m|p|ə|l| |d|ˈɑː|t| |k|ˈɑː|m| |s|l|ˈæ|ʃ| |p|ˈæ|θ| |f|ɔː|ɹ| |d|iː|t|ˈe|ɪ|l|z|."

    def test_en_backslash_path(self):
        assert self._p("C:\\Users\\name\\file.txt was found.", "en-us") == "s|ˈiː|:|b|ˈæ|k|s|l|æ|ʃ| |j|ˈuː|z|ɚ|z| |b|ˈæ|k|s|l|æ|ʃ| |n|ˈe|ɪ|m| |b|ˈæ|k|s|l|æ|ʃ| |f|ˈa|ɪ|l|.|t|ˌiː|ˌɛ|k|s|t|ˈiː| |w|ʌ|z| |f|ˈa|ʊ|n|d|."

    def test_en_brackets(self):
        assert self._p("The value is <10> or (20) or [30] or {40}.", "en-us") == "ð|ə| |v|ˈæ|l|j|uː| |ɪ|z|,| |t|ˈɛ|n|,| |ɔː|ɹ|,| |t|w|ˈɛ|n|t|i|,| |ɔː|ɹ|,| |θ|ˈɜː|ɾ|i|,| |ɔː|ɹ|,| |f|ˈɔː|ɹ|ɾ|i|."

    def test_en_em_dash(self):
        assert self._p("It was a dark and stormy night—except at occasional intervals.", "en-us") == "ɪ|t| |w|ʌ|z|ɐ| |d|ˈɑː|ɹ|k| |æ|n|d| |s|t|ˈoː|ɹ|m|i| |n|ˈa|ɪ|t|,| |ɛ|k|s|ˈɛ|p|t| |æ|ɾ| |ə|k|ˈe|ɪ|ʒ|ə|n|ə|l| |ˈɪ|n|t|ɚ|v|ə|l|z|."

    def test_fr_guillemets_removed(self):
        assert self._p("Elle a dit «bonjour» à lui.", "fr-fr") == "ɛ|l| |a| |d|ˈi| |b|ɔ̃|ʒ|ˈu|ʁ| |a| |l|y|ˈi|."

    def test_fr_em_dash(self):
        assert self._p("La pluie tombait à torrents—sauf à intervalles occasionnels.", "fr-fr") == "l|a|-| |p|l|y|ˈi| |t|ɔ̃|b|ˈɛ|t| |a| |t|o|ʁ|ˈɑ̃|,| |s|ˈo|f| |a| |ɛ̃|t|ɛ|ʁ|v|ˈa|l|z| |ɔ|k|a|z|j|ɔ|n|ˈɛ|l|."

    def test_ro_question(self):
        assert self._p("Oare?", "ro") == "ˈɔ|a|ɾ|e|?"

    def test_ro_exclamation(self):
        assert self._p("Doare!", "ro") == "d|ˈɔ|a|ɾ|e|!"

    def test_ro_hyphen(self):
        assert self._p("N-are.", "ro") == "n|ˈa|ɾ|e|."

    def test_ro_trailing_whitespace(self):
        assert self._p("Cuvânt   ", "ro") == "k|u|v|ˈɨ|n|t|."

    def test_ro_em_dash(self):
        assert self._p("Ploaia cădea în torente—cu excepția momentelor ocazionale.", "ro") == "p|l|ˈɔ|a|j|a| |k|ə|d|ˈe|a| |ɨ|n| |t|o|ɾ|ˈe|n|t|e|,| |k|u| |e|k|s|t|ʃ|ˈe|p|t|s|j|a| |m|ˌo|m|e|n|t|ˈe|l|o|r| |ˌo|k|a|z|j|o|n|ˈa|l|e|."

    # --- missing EN cases from original main() ---

    def test_en_percent(self):
        assert self._p("He scored 95% on the test.", "en-us") == "h|iː| |s|k|ˈoː|ɹ|d| |n|ˈa|ɪ|n|t|i| |f|ˈa|ɪ|v| |p|ɚ|s|ˈɛ|n|t| |ɔ|n|ð|ə| |t|ˈɛ|s|t|."

    def test_en_phone_address(self):
        assert self._p("Call me at 555-1234 or visit 123 Main St.", "en-us") == "k|ˈɔː|l| |m|ˌiː| |æ|t| |f|ˈa|ɪ|v| |h|ˈʌ|n|d|ɹ|ɪ|d| |æ|n|d| |f|ˈɪ|f|t|i| |f|ˈa|ɪ|v| |t|w|ˈɛ|l|v| |θ|ˈɜː|ɾ|i| |f|ˈoː|ɹ| |ɔː|ɹ| |v|ˈɪ|z|ɪ|t| |w|ˈʌ|n| |t|w|ˈɛ|n|t|i| |θ|ɹ|ˈiː| |m|ˈe|ɪ|n| |s|t|ɹ|ˈiː|t|."

    def test_en_years_en_dash(self):
        assert self._p("The years 2020—2025 were challenging.", "en-us") == "ð|ə| |j|ˈɪ|ɹ|z| |t|w|ˈɛ|n|t|i| |t|w|ˈɛ|n|t|i|,| |t|w|ˈɛ|n|t|i| |t|w|ˈɛ|n|t|i| |f|ˈa|ɪ|v| |w|ɜː| |t|ʃ|ˈæ|l|ə|n|d|ʒ|ˌɪ|ŋ|."

    def test_en_smart_quotes_and_right_single(self):
        assert self._p("He said “hello” to me and I've said ‘hello’ back.", "en-us") == "h|iː| |s|ˈɛ|d| |h|ə|l|ˈo|ʊ| |t|ə| |m|ˌiː| |æ|n|d| |a|ɪ|v| |s|ˈɛ|d| |h|ə|l|ˈo|ʊ| |b|ˈæ|k|."

    # --- RO missing whitespace variants ---

    def test_ro_trailing_newlines(self):
        assert self._p("Cuvânt\n\n", "ro") == "k|u|v|ˈɨ|n|t|."

    def test_ro_trailing_tab(self):
        assert self._p("Cuvânt\t", "ro") == "k|u|v|ˈɨ|n|t|."

    # --- ES phonemizer ---

    def test_es_doctor(self):
        assert self._p("El Dr. García llegará a las 15:00.", "es") == "e|l| |ð|o|k|t|ˈo|ɾ| |ɣ|a|ɾ|θ|ˈi|a| |ʎ|ˌe|ɣ|a|ɾ|ˈa| |a| |l|a|s| |k|ˈi|n|θ|e|."

    def test_es_precio(self):
        # NeMo ES does not handle $5.00 well - documents current behavior
        assert self._p("El precio es $5.00 desde el 21 de enero de 2026.", "es") == "e|l| |p|ɾ|ˈe|θ|j|o| |ˈe|s| |s|ˈi|ɡ|n|o| |ð|e| |ð|ˈo|l|a|ɾ| |θ|ˈi|n|k|o| |p|ˈu|n|t|o| |θ|ˈe|ɾ|o| |θ|ˈe|ɾ|o| |ð|ˌe|s|ð|e| |e|l| |β|e|ɪ|n|t|j|ˈu|n|o| |ð|e| |e|n|ˈe|ɾ|o| |ð|e| |ð|ˈo|s| |m|ˈi|l| |β|ˌe|ɪ|n|t|i|s|ˈe|i|s|."

    def test_es_temperatura(self):
        assert self._p("La temperatura es -5°C o 23°F.", "es") == "l|a| |t|ˌe|m|p|e|ɾ|a|t|ˈu|ɾ|a| |ˈe|s| |θ|ˈi|n|k|o| |ɣ|ɾ|ˈa|ð|o|s| |θ|ˈe| |o| |β|ˌe|ɪ|n|t|i|t|ɾ|ˈe|s| |ɣ|ɾ|ˈa|ð|o|s| |ˈɛ|f|e|."

    # --- DE phonemizer ---

    def test_de_doctor(self):
        assert self._p("Dr. M\u00fcller sieht Sie um 15:00 Uhr.", "de") == "d|ˈɔ|k|t|oː|ɾ| |m|ˈy|l|ɜ| |z|ˈiː|t| |z|iː| |ʊ|m| |f|ˈy|n|f|t|s|eː|n| |ˈuː|ɾ|."

    def test_de_preis(self):
        assert self._p("Der Preis beträgt 5,00€ ab dem 21. Januar 2026.", "de") == "d|ɛ|ɾ| |p|ɾ|ˈa|ɪ|s| |b|ə|t|ɾ|ˈɛː|k|t| |f|ˈy|n|f|,| |n|ˈʊ|l| |n|ˈʊ|l| |ˈɔ|ø|r|oː| |a|p| |d|eː|m| |a|ɪ|n| |ʊ|n|t| |t|s|v|ˈa|n|t|s|ɪ|ç|s|t|ɜ| |j|ˈa|n|uː|ˌɑː|ɾ| |t|s|v|ˈa|ɪ| |t|ˈa|ʊ|z|ə|n|t| |z|ˈɛ|k|s| |ʊ|n|t| |t|s|v|ˈa|n|t|s|ɪ|ç|s|t|ə|."

    def test_de_temperatur(self):
        assert self._p("Die Temperatur beträgt -5°C oder 23°F.", "de") == "d|iː| |t|ˌɛ|m|p|eː|r|a|t|ˈuː|ɾ| |b|ə|t|ɾ|ˈɛː|k|t| |m|ˈiː|n|ʊ|s| |f|ˈy|n|f| |ɡ|ɾ|ˈɑː|t| |t|s|ˈɛ|l|z|iː|ˌʊ|s| |ˌoː|d|ɜ| |d|ɾ|ˈa|ɪ| |ʊ|n|t| |t|s|v|ˈa|n|t|s|ɪ|ç| |ɡ|ɾ|ˈɑː|t| |f|ˈɑː|r|ə|n|h|ˌa|ɪ|t|."

    # --- IT phonemizer ---

    def test_it_doctor(self):
        assert self._p("Il Dr. Rossi la vedrà alle 15:00.", "it") == "i|l| |d|o|tː|ˈɔ|r| |r|ˈo|s|s|ɪ| |l|a| |v|e|d|r|ˈa| |ˌa|l|l|e| |k|w|ˈi|n|d|i|t|ʃ|ɪ|."

    def test_it_price(self):
        assert self._p("Il prezzo è €5,00 dal 21 gennaio 2026.", "it") == "i|l| |p|r|ˈɛ|t|sː|o| |e| |t|ʃ|ˈi|n|k|w|e| |ˈɛ|ʊ|r|o| |d|z|ˈɛ|r|o| |d|z|ˈɛ|r|o| |d|a|l| |v|e|n|t|ˈu|n|o| |d|ʒ|e|n|n|ˈa|i|o| |d|ʊ|e|m|ˈi|l|a| |v|e|n|t|ɪ|s|ˈɛ|j|."

    def test_it_temperature(self):
        assert self._p("La temperatura è -5°C o 23°F.", "it") == "l|a| |t|e|m|p|e|r|a|t|ˈu|r|a| |e| |m|ˈe|n|o| |t|ʃ|ˈi|n|k|w|e| |ɡ|r|ˈa|d|o| |t|s|e|l|s|j|ˈu|s| |o| |v|e|n|t|i|t|r|ˈe| |ɡ|r|ˈa|d|o| |f|ˈa|r|e|n|a|ɪ|t|."

    # --- PT phonemizer (no Nemo, eSpeak only) ---

    def test_pt_doctor(self):
        assert self._p("O Dr. Silva verá você às 15:00.", "pt") == "ʊ| |d|o|w|t|ˈo|r|.| |s|ˈi|l|v|ɐ| |v|ɨ|ɾ|ˈa| |v|o|s|ˌe| |ɐ|ɐ|ʃ| |k|ˈi|ŋ|z|ɨ|:|z|ˈɛ|ɾ|u| |z|ˈɛ|ɾ|u|."

    def test_pt_price(self):
        assert self._p("O preço é R$ 5,00 desde 21 de janeiro de 2026.", "pt") == "ʊ| |p|ɹ|ˈe|s|w| |ɛ| |ʁ|ɨ|ˈa|ʊ| |s|ˈi|ŋ|k|u|,| |z|ˈɛ|ɾ|u| |z|ˈɛ|ɾ|u| |d|ˈe|ʒ|d|ɨ| |v|ˈi|ŋ|t|ɨ|i|ˈu|m| |d|ɨ| |ʒ|ɐ|n|ˈe|ɪ|ɾ|ʊ| |d|ɨ| |d|ˈo|ɪ|ʒ| |m|ˈi|l| |i| |v|ˈi|ŋ|t|ɨ|i|s|ˈe|ɪ|ʃ|."

    def test_pt_temperature(self):
        assert self._p("A temperatura é -5°C ou 23°F.", "pt") == "ɐ| |t|ˌe|ɪ|m|p|ɨ|ɾ|ɐ|t|ˈu|ɾ|ɐ| |ɛ| |m|ˈe|n|ʊ|s| |s|ˈi|ŋ|k|u| |ɡ|ɹ|ˈa|ʊ| |s|ˈe| |ˈo|w| |v|ˈi|ŋ|t|ɨ|i|t|ɹ|ˈe|ʒ| |ɡ|ɹ|ˈa|ʊ| |ˈɛ|f|."

    def test_en_syllabic_n(self):
        # This specific sentence triggers a syllabic n (ˌn̩) in eSpeak output.
        # validate_group must keep ˌn̩ as a single token since it is in symbols.
        result = self._p("He unfolded a long typewritten letter, and handed it to Gregson.", "en-us")
        assert "ˌn̩" in result.split("|")

    def test_fr_semivowel_stress(self):
        # This specific sentence triggers a stressed semivowel (ˈw) in eSpeak output.
        # validate_group must keep ˈw as a single token since it is in symbols.
        result = self._p("Le sens de l'ouïe est plus fin que celui du toucher.", "fr-fr")
        assert "ˈw" in result.split("|")

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
        result = multilingual_phonemizer(_punctuation, "en-us")
        surviving = set(result) & set(_punctuation)
        assert surviving == set(' ;:,.!?'), f"Unexpected surviving punctuation: {surviving}"

class TestValidateGroup:
    """Tests for the validate_group safety check in split_ipa."""

    def test_unknown_group_is_split(self):
        from matcha.text.phonemizers import split_ipa
        from matcha.text.symbols import symbol_to_id
        # ˈb is not in symbols, because stress cannot precede a plain consonant, so we expect split_ipa to 
        # to split the group and keep individual phonemes.
        tokens = split_ipa("ˈb")
        assert tokens == ["ˈ", "b"]
        assert symbol_to_id.get("ˈb") is None
