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
        assert cleanup_text("The value is (20).") == "The value is, 20."

    def test_square_brackets_become_comma(self):
        assert cleanup_text("The value is [30].") == "The value is, 30."

    def test_curly_brackets_become_comma(self):
        assert cleanup_text("The value is {40}.") == "The value is, 40."

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
        return multilingual_phonemizer(text, lang).strip()

    def test_en_plain(self):
        assert self._p("I live for live broadcasts.", "en-us") == "aɪ lˈɪv fɔːɹ lˈaɪv bɹˈɔːdkæsts."

    def test_en_doctor(self):
        assert self._p("Dr. Jones will see you at 15:00.", "en-us") == "dˈɑːktɚ dʒˈoʊnz wɪl sˈiː juː æt fˈɪftiːn əklˈɑːk."

    def test_en_price(self):
        assert self._p("The price is $5.00 as of Jan 21st, 2026.", "en-us") == "ðə pɹˈaɪs ɪz fˈaɪv dˈɑːlɚz æz ʌv dʒˈænjuːˌɛɹi twˈɛnti fˈɜːst, twˈɛnti twˈɛnti sˈɪks."

    def test_en_temperature(self):
        assert self._p("The temperature is -5°C or 23°F.", "en-us") == "ðə tˈɛmpɹɪtʃɚɹ ɪz mˈaɪnəs fˈaɪv dᵻɡɹˈiːz sˈɛlsɪəs ɔːɹ twˈɛnti θɹˈiː dᵻɡɹˈiːz fˈæɹənhˌaɪt."

    def test_en_ellipsis(self):
        assert self._p("He thought… and then spoke.", "en-us") == "hiː θˈɔːt, ænd ðˈɛn spˈoʊk."

    def test_en_url(self):
        assert self._p("Visit http://example.com/path for details.", "en-us") == "vˈɪzɪt ˌeɪtʃtˌiːtˌiːpˈiː kˈoʊlən slˈæʃ slˈæʃ ɛɡzˈæmpəl dˈɑːt kˈɑːm slˈæʃ pˈæθ fɔːɹ diːtˈeɪlz."

    def test_en_backslash_path(self):
        assert self._p("C:\\Users\\name\\file.txt was found.", "en-us") == "sˈiː:bˈækslæʃ jˈuːzɚz bˈækslæʃ nˈeɪm bˈækslæʃ fˈaɪl.tˌiːˌɛkstˈiː wʌz fˈaʊnd."

    def test_en_brackets(self):
        assert self._p("The value is <10> or (20) or [30] or {40}.", "en-us") == "ðə vˈæljuː ɪz, tˈɛn, ɔːɹ, twˈɛnti, ɔːɹ, θˈɜːɾi, ɔːɹ, fˈɔːɹɾi."

    def test_en_em_dash(self):
        assert self._p("It was a dark and stormy night—except at occasional intervals.", "en-us") == "ɪt wʌzɐ dˈɑːɹk ænd stˈoːɹmi nˈaɪt, ɛksˈɛpt æɾ əkˈeɪʒənəl ˈɪntɚvəlz."

    def test_fr_guillemets_removed(self):
        assert self._p("Elle a dit «bonjour» à lui.", "fr-fr") == "ɛl a dˈi bɔ̃ʒˈuʁ a lyˈi."

    def test_fr_em_dash(self):
        assert self._p("La pluie tombait à torrents—sauf à intervalles occasionnels.", "fr-fr") == "la- plyˈi tɔ̃bˈɛt a toʁˈɑ̃, sˈof a ɛ̃tɛʁvˈalz ɔkazjɔnˈɛl."

    def test_ro_question(self):
        assert self._p("Oare?", "ro") == "ˈɔaɾe?"

    def test_ro_exclamation(self):
        assert self._p("Doare!", "ro") == "dˈɔaɾe!"

    def test_ro_hyphen(self):
        assert self._p("N-are.", "ro") == "nˈaɾe."

    def test_ro_trailing_whitespace(self):
        assert self._p("Cuvânt   ", "ro") == "kuvˈɨnt."

    def test_ro_em_dash(self):
        assert self._p("Ploaia cădea în torente—cu excepția momentelor ocazionale.", "ro") == "plˈɔaja kədˈea ɨn toɾˈente, ku ekstʃˈeptsja mˌomentˈelor ˌokazjonˈale."
