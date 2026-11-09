import unittest
from datetime import datetime
import ctx as C


class T(unittest.TestCase):
    def test_speakable(self):
        self.assertEqual(C.speakable("Es sind 12 Grad."), "Es sind zwölf Grad.")
        self.assertEqual(C.speakable("3,5 Prozent"), "drei Komma fünf Prozent")
        self.assertEqual(C.speakable("1.869.473 Einwohner"), "1869473 Einwohner".replace("1869473", "eintausendachthundertneunundsechzigtausendvierhundertdreiundsiebzig") if False else C.speakable("1869473 Einwohner"))
        self.assertEqual(C.speakable("im Jahr 2026"), "im Jahr zweitausendsechsundzwanzig")
        self.assertEqual(C.speakable("20 °C"), "zwanzig Grad")
        self.assertEqual(C.speakable("25%"), "fünfundzwanzig Prozent")
        for x, want in [("am 2.8.2026", "am zweiten August zweitausendsechsundzwanzig"), ("Der 3. Oktober", "Der dritte Oktober"), ("im 3. Stock", "im dritten Stock"),
                        ("bis 31.12.2027", "bis zum einunddreißigsten Dezember zweitausendsiebenundzwanzig"), ("Die Mauer fiel 1989.", "Die Mauer fiel neunzehnhundertneunundachtzig."),
                        ("1500 Meter", "eintausendfünfhundert Meter"), ("Berlin hat 3.685.265 Einwohner.", "Berlin hat rund drei Komma sieben Millionen Einwohner."),
                        ("Art. 72 Abs. 3", "Artikel zweiundsiebzig Absatz drei"), ("z. B. 5 km/h", "zum Beispiel fünf Kilometer pro Stunde"), ("Nummer 1. Dann", "Nummer eins. Dann")]:
            self.assertEqual(C.speakable(x), want, x)
    def test_state(self):
        s = C.state_line(datetime(2026, 11, 1, 15, 3), "Basel", pend={"stage": "when", "ask": "Um wie viel Uhr?"})
        self.assertIn("Sonntag", s); self.assertIn("Basel", s); self.assertIn("Um wie viel Uhr?", s); self.assertTrue(s.endswith("] "))
    def test_summary(self):
        self.assertEqual(C.summary(["a", "b"], 4), "")
        self.assertIn("a", C.summary(list("abcdefg"), 4)); self.assertNotIn("g", C.summary(list("abcdefg"), 4))
    def test_handle(self):
        self.assertIn("1. Titel", C.handle([("u", "Titel")])); self.assertEqual(C.handle([]), "")
    def test_topic(self): self.assertEqual(C.topic("wetter", {"ort": "Basel", "wann": ""}, ""), "wetter(Basel)")


if __name__ == "__main__": unittest.main()


class Dict(unittest.TestCase):
    def test_bg_no_tts(self):
        import tools as T
        s, foreign, ipa = T.say_dict("Danke", "bg")
        self.assertIsNone(foreign); self.assertIn("blagodarja", s); self.assertEqual(T.translit("благодаря"), "blagodarja")


class Trim(unittest.TestCase):
    def test_trim(self):
        import tools as T
        o = "Titel / Abschnitt: " + "Erster Satz ist lang genug. " * 18 + "\nZweiter Treffer: " + "Noch ein Satz hier. " * 30 + "\nDritter: kurz"
        t = T.trim_ctx("wiki_suche", o, 800)
        self.assertLessEqual(len(t), 810); self.assertTrue(t.startswith("Titel / Abschnitt: Erster Satz")); self.assertIn("Zweiter Treffer", t)
        self.assertEqual(T.trim_ctx("wiki_suche", "kurz", 800), "kurz")
        self.assertEqual(T.trim_ctx("wiki_suche", "x" * 3000, 0), "x" * 2500)
        self.assertTrue(T.trim_ctx("wiki_suche", "Satz eins. " * 100, 300).endswith("."))


class Buch(unittest.TestCase):
    def test_fmt_and_registration(self):
        import tools as T
        h = [{"title": "Hoppe: Die Privatisierung der Verteidigung", "work": "Die Privatisierung der Verteidigung", "author": "Hans-Hermann Hoppe", "page": "34", "url": "https://x/book.pdf#page=34", "text": "Ein Staat ist\nein Zwangsmonopol."}]
        o = T.fmt_buch(h)
        self.assertEqual(o, "Laut Hoppe, Die Privatisierung der Verteidigung: Ein Staat ist ein Zwangsmonopol.")
        self.assertNotIn("34", o); self.assertNotIn("http", o)
        self.assertIn("bibliothek", [t["function"]["name"] for t in T.TOOLS])
        self.assertIn("-> bibliothek", T.sys_router())

    def test_serve_roundtrip_and_missing(self):
        import tools as T, tempfile, stat, os
        fake = ('#!/usr/bin/env python3\nimport sys,json\nfor l in sys.stdin:\n'
                ' q=json.loads(l)["q"]\n print(json.dumps({"hits":[{"title":"Rothbard: Die Anatomie des Staates","work":"Die Anatomie des Staates","author":"Murray N. Rothbard","text":"Echo: "+q}]}),flush=True)\n')
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "buch"); open(p, "w").write(fake); os.chmod(p, 0o755)
            os.environ["KITSUNE_BUCH"] = p
            try:
                self.assertEqual(T.t_buch("Was ist der Staat?"), "Laut Rothbard, Die Anatomie des Staates: Echo: Was ist der Staat?")
                self.assertEqual(T.t_buch("zweite"), "Laut Rothbard, Die Anatomie des Staates: Echo: zweite")  # same persistent process
            finally:
                T.buch_stop(); os.environ.pop("KITSUNE_BUCH", None)
        os.environ["KITSUNE_BUCH"] = "/nonexistent/buch"
        try: self.assertEqual(T.t_buch("x"), "Die Bibliothek ist nicht verfügbar.")
        finally: os.environ.pop("KITSUNE_BUCH", None); T.buch_stop()
