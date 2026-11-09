import unittest
from datetime import date as D, datetime as DT, time as T
from dates import resolve, speak, kal_at

SUN = DT(2026, 11, 1, 10, 0)   # Sunday
TUE = DT(2026, 11, 3, 16, 0)   # Tuesday


class Base(unittest.TestCase):
    def chk(self, phrase, d, t=None, now=SUN, allday=False):
        r = resolve(phrase, now)
        self.assertTrue(r.ok, (phrase, r))
        self.assertEqual((r.date, r.time, r.allday), (D(*d) if d else None, T(*t) if t else None, allday), phrase)
        return r


class TestDays(Base):
    def test_heute(self): self.chk("heute", (2026, 11, 1))
    def test_morgen(self): self.chk("morgen", (2026, 11, 2))
    def test_uebermorgen(self): self.chk("übermorgen", (2026, 11, 3))
    def test_uebermorgen_ascii(self): self.chk("Uebermorgen", (2026, 11, 3))
    def test_gestern(self):
        r = self.chk("gestern", (2026, 10, 31))
        self.assertIn("Vergangenheit", r.note)
    def test_case_punct(self): self.chk("MORGEN, um 14:30 Uhr.", (2026, 11, 2), (14, 30))
    def test_in_tagen(self): self.chk("in drei Tagen", (2026, 11, 4))
    def test_in_tagen_digit(self): self.chk("in 10 Tagen", (2026, 11, 11))
    def test_in_woche(self): self.chk("in einer Woche", (2026, 11, 8))
    def test_in_wochen(self): self.chk("in 2 Wochen", (2026, 11, 15))
    def test_in_monaten(self): self.chk("in 3 Monaten", (2027, 2, 1))
    def test_in_tagen_um(self): self.chk("in zwei Tagen um 9 Uhr", (2026, 11, 3), (9, 0))


class TestRelTime(Base):
    def test_hours(self): self.chk("in 2 Stunden", (2026, 11, 1), (12, 0))
    def test_minutes(self): self.chk("in 30 Minuten", (2026, 11, 1), (10, 30))
    def test_word_hours(self): self.chk("in zwei Stunden", (2026, 11, 3), (18, 0), now=TUE)
    def test_half_hour(self): self.chk("in einer halben Stunde", (2026, 11, 1), (10, 30))
    def test_cross_midnight(self): self.chk("in 9 Stunden", (2026, 11, 4), (1, 0), now=TUE)
    def test_one_hour(self): self.chk("in einer Stunde", (2026, 11, 1), (11, 0))


class TestWeekdays(Base):
    def test_bare(self): self.chk("am Montag", (2026, 11, 2))
    def test_bare_same_day_no_time(self): self.chk("Sonntag", (2026, 11, 8))
    def test_bare_same_day_ahead(self): self.chk("Sonntag um 14 Uhr", (2026, 11, 1), (14, 0))
    def test_bare_same_day_passed(self): self.chk("Sonntag um 9 Uhr", (2026, 11, 8), (9, 0))
    def test_tue_ahead(self): self.chk("am Dienstag um 17 Uhr", (2026, 11, 3), (17, 0), now=TUE)
    def test_tue_passed(self): self.chk("am Dienstag um 15 Uhr", (2026, 11, 10), (15, 0), now=TUE)
    def test_naechsten(self): self.chk("nächsten Montag", (2026, 11, 2))
    def test_naechsten_same_day(self): self.chk("nächsten Dienstag", (2026, 11, 10), now=TUE)
    def test_kommenden(self): self.chk("kommenden Freitag", (2026, 11, 6))
    def test_kommenden_tue(self): self.chk("kommenden Freitag", (2026, 11, 6), now=TUE)
    def test_uebernaechsten(self): self.chk("übernächsten Freitag", (2026, 11, 13))
    def test_in_a_week_sun(self): self.chk("Montag in einer Woche", (2026, 11, 9))
    def test_in_a_week_tue(self): self.chk("Montag in einer Woche", (2026, 11, 16), now=TUE)
    def test_same_wd_in_a_week(self): self.chk("Dienstag in einer Woche", (2026, 11, 10), now=TUE)
    def test_naechste_woche(self): self.chk("nächste Woche Mittwoch", (2026, 11, 11), now=TUE)
    def test_weekday_time(self): self.chk("Montag um halb drei", (2026, 11, 2), (14, 30))
    def test_sonnabend(self): self.chk("Sonnabend", (2026, 11, 7))
    def test_freitagabend(self): self.chk("Freitagabend", (2026, 11, 6), (19, 0))
    def test_freitag_abend(self): self.chk("am Freitag Abend", (2026, 11, 6), (19, 0))
    def test_montag_morgen(self): self.chk("Montag morgen", (2026, 11, 2), (8, 0))
    def test_freitag_today_abend(self):
        self.chk("Freitagabend", (2026, 11, 6), (19, 0), now=DT(2026, 11, 6, 9, 0))


class TestDates(Base):
    def test_dom_this_month(self): self.chk("am 5.", (2026, 11, 5))
    def test_dom_today(self): self.chk("am 1.", (2026, 11, 1))
    def test_dom_next_month(self): self.chk("am 1.", (2026, 12, 1), now=TUE)
    def test_dom_31_skip(self): self.chk("am 31.", (2026, 12, 31))
    def test_dom_word(self): self.chk("am fünften", (2026, 11, 5))
    def test_day_month(self): self.chk("am 5. November", (2026, 11, 5))
    def test_day_month_next_year(self): self.chk("am 5. Oktober", (2027, 10, 5))
    def test_numeric(self): self.chk("am 5.11.", (2026, 11, 5))
    def test_numeric_year(self): self.chk("5.11.2027", (2027, 11, 5))
    def test_numeric_year2(self): self.chk("am 5.11.27", (2027, 11, 5))
    def test_numeric_with_time(self): self.chk("am 5.11. um 14 Uhr", (2026, 11, 5), (14, 0))
    def test_iso(self): self.chk("2026-12-24", (2026, 12, 24))
    def test_ordinal_word(self): self.chk("dritten Dezember", (2026, 12, 3))
    def test_ordinal_maerz(self): self.chk("am fünften März", (2027, 3, 5))
    def test_ordinal_21(self): self.chk("am einundzwanzigsten Dezember", (2026, 12, 21))
    def test_ordinal_achten(self): self.chk("am achten Januar", (2027, 1, 8))
    def test_ordinal_ersten(self): self.chk("am ersten Dezember", (2026, 12, 1))
    def test_leap(self): self.chk("am 29. Februar", (2028, 2, 29))
    def test_digit_month_year(self): self.chk("am 5. November 2027", (2027, 11, 5))
    def test_past_date_note(self):
        r = self.chk("am 1.10.2026", (2026, 10, 1))
        self.assertIn("Vergangenheit", r.note)
    def test_invalid(self):
        r = resolve("am 31.2.2027", SUN)
        self.assertFalse(r.ok)


class TestTimes(Base):
    def test_uhr(self): self.chk("heute um 14 Uhr", (2026, 11, 1), (14, 0))
    def test_colon(self): self.chk("morgen 14:30", (2026, 11, 2), (14, 30))
    def test_uhr_min(self): self.chk("morgen 14 Uhr 30", (2026, 11, 2), (14, 30))
    def test_words_uhr(self): self.chk("morgen um vierzehn Uhr dreißig", (2026, 11, 2), (14, 30))
    def test_halb_drei(self): self.chk("heute um halb drei", (2026, 11, 1), (14, 30))
    def test_viertel_nach(self): self.chk("morgen viertel nach vier", (2026, 11, 2), (16, 15))
    def test_viertel_vor(self): self.chk("morgen viertel vor fünf", (2026, 11, 2), (16, 45))
    def test_dreiviertel(self): self.chk("morgen dreiviertel fünf", (2026, 11, 2), (16, 45))
    def test_zehn_nach(self): self.chk("morgen zehn nach drei", (2026, 11, 2), (15, 10))
    def test_zwanzig_vor(self): self.chk("morgen zwanzig vor acht", (2026, 11, 2), (7, 40))
    def test_fuenf_vor_halb(self): self.chk("morgen fünf vor halb zwei", (2026, 11, 2), (13, 25))
    def test_halb_eins(self): self.chk("morgen halb eins", (2026, 11, 2), (12, 30))
    def test_halb_zwoelf(self): self.chk("morgen halb zwölf", (2026, 11, 2), (11, 30))
    def test_halb_sieben(self): self.chk("morgen halb sieben", (2026, 11, 2), (18, 30))
    def test_halb_acht(self): self.chk("morgen halb acht", (2026, 11, 2), (7, 30))
    def test_um_acht(self): self.chk("morgen um acht", (2026, 11, 2), (8, 0))
    def test_um_zwoelf(self): self.chk("morgen um zwölf", (2026, 11, 2), (12, 0))
    def test_um_fuenf(self): self.chk("morgen um 5", (2026, 11, 2), (17, 0))
    def test_zero(self): self.chk("morgen um 0 Uhr", (2026, 11, 2), (0, 0))
    def test_padded_24h(self): self.chk("morgen um 08:00", (2026, 11, 2), (8, 0))
    def test_padded_explicit(self): self.chk("morgen um 03:00", (2026, 11, 2), (3, 0))
    def test_unpadded_colon_pm(self): self.chk("morgen 3:30", (2026, 11, 2), (15, 30))
    def test_acht_abends(self): self.chk("morgen acht Uhr abends", (2026, 11, 2), (20, 0))
    def test_sieben_morgens(self): self.chk("morgen um 7 Uhr morgens", (2026, 11, 2), (7, 0))
    def test_vier_nachmittags(self): self.chk("morgen um vier nachmittags", (2026, 11, 2), (16, 0))
    def test_drei_nachts(self): self.chk("morgen um drei Uhr nachts", (2026, 11, 2), (3, 0))
    def test_zehn_nachts(self): self.chk("morgen um zehn nachts", (2026, 11, 2), (22, 0))
    def test_eins_mittags(self): self.chk("morgen um eins mittags", (2026, 11, 2), (13, 0))
    def test_mitternacht(self): self.chk("Mitternacht", (2026, 11, 2), (0, 0))
    def test_samstag_mitternacht(self): self.chk("Samstag Mitternacht", (2026, 11, 8), (0, 0))
    def test_mittags(self): self.chk("morgen mittags", (2026, 11, 2), (12, 0))
    def test_morgens(self): self.chk("übermorgen morgens", (2026, 11, 3), (8, 0))
    def test_vormittags(self): self.chk("morgen vormittags", (2026, 11, 2), (10, 0))
    def test_nachmittags(self): self.chk("morgen nachmittags", (2026, 11, 2), (15, 0))
    def test_abends(self): self.chk("morgen abends", (2026, 11, 2), (19, 0))
    def test_nachts(self): self.chk("morgen nachts", (2026, 11, 2), (22, 0))
    def test_heute_abend(self): self.chk("heute Abend", (2026, 11, 1), (19, 0))
    def test_heute_morgen_is_daypart(self):
        self.chk("heute morgen", (2026, 11, 1), (8, 0), now=DT(2026, 11, 1, 6, 0))
    def test_morgen_frueh(self): self.chk("morgen früh", (2026, 11, 2), (8, 0))
    def test_morgen_morgen(self): self.chk("morgen Morgen", (2026, 11, 2), (8, 0))
    def test_morgen_abend(self): self.chk("morgen Abend", (2026, 11, 2), (19, 0))
    def test_frueh_hour(self): self.chk("morgen um 6 Uhr früh", (2026, 11, 2), (6, 0))
    def test_ue_spelling(self): self.chk("uebermorgen um fuenf vor halb zwei", (2026, 11, 3), (13, 25))


class TestRoll(Base):
    def test_time_ahead_today(self): self.chk("um 14 Uhr", (2026, 11, 1), (14, 0))
    def test_time_passed_rolls(self):
        r = self.chk("um 9 Uhr", (2026, 11, 2), (9, 0))
        self.assertIn("nächster Tag", r.note)
    def test_tue_halb_drei_rolls(self): self.chk("um halb drei", (2026, 11, 4), (14, 30), now=TUE)
    def test_tue_um_acht_rolls(self): self.chk("um acht", (2026, 11, 4), (8, 0), now=TUE)
    def test_explicit_today_not_rolled(self):
        r = self.chk("heute um 9 Uhr", (2026, 11, 1), (9, 0))
        self.assertIn("Vergangenheit", r.note)
    def test_abends_today(self): self.chk("abends", (2026, 11, 1), (19, 0))
    def test_morgens_rolls(self): self.chk("morgens", (2026, 11, 2), (8, 0))


class TestAllday(Base):
    def test_ganztaegig(self): self.chk("morgen ganztägig", (2026, 11, 2), allday=True)
    def test_ganztags(self): self.chk("am 5. November ganztags", (2026, 11, 5), allday=True)
    def test_date_only_not_allday(self):
        r = self.chk("morgen", (2026, 11, 2))
        self.assertFalse(r.allday)
        self.assertIn("keine Uhrzeit", r.note)


class TestBad(Base):
    def test_empty(self): self.assertFalse(resolve("", SUN).ok)
    def test_garbage(self): self.assertFalse(resolve("Das Wetter ist schön", SUN).ok)
    def test_bad_result_fields(self):
        r = resolve("blabla", SUN)
        self.assertIsNone(r.date); self.assertIsNone(r.time)
        self.assertEqual(kal_at(r), "")
        self.assertIn("nicht verstanden", speak(r, SUN))


class TestOutput(Base):
    def test_speak_tomorrow(self):
        self.assertEqual(speak(resolve("morgen um 14:30", SUN), SUN), "morgen, Montag den 2. November, um 14 Uhr 30")
    def test_speak_today_halb(self):
        self.assertEqual(speak(resolve("heute um halb drei", SUN), SUN), "heute um halb drei")
    def test_speak_today_full_hour(self):
        self.assertEqual(speak(resolve("heute um 14 Uhr", SUN), SUN), "heute um 14 Uhr")
    def test_speak_today_only(self):
        self.assertEqual(speak(resolve("heute", SUN), SUN), "heute")
    def test_speak_viertel_today(self):
        self.assertEqual(speak(resolve("heute viertel vor fünf", SUN), SUN), "heute um viertel vor fünf")
    def test_speak_day_after(self):
        self.assertEqual(speak(resolve("übermorgen", SUN), SUN), "übermorgen, Dienstag den 3. November")
    def test_speak_far(self):
        self.assertEqual(speak(resolve("nächsten Freitag um 9 Uhr", SUN), SUN), "Freitag den 6. November, um 9 Uhr")
    def test_speak_next_year(self):
        self.assertEqual(speak(resolve("5.1.2027", SUN), SUN), "Dienstag den 5. Januar 2027")
    def test_speak_allday(self):
        self.assertEqual(speak(resolve("morgen ganztägig", SUN), SUN), "morgen, Montag den 2. November, ganztägig")
    def test_speak_midnight(self):
        self.assertIn("um Mitternacht", speak(resolve("Mitternacht", SUN), SUN))
    def test_kal_time(self): self.assertEqual(kal_at(resolve("morgen 14:30", SUN)), "2026-11-02 14:30")
    def test_kal_allday(self): self.assertEqual(kal_at(resolve("morgen ganztags", SUN)), "2026-11-02")
    def test_kal_date_only(self): self.assertEqual(kal_at(resolve("morgen", SUN)), "2026-11-02")


if __name__ == "__main__":
    unittest.main()
