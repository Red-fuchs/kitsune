import unittest, rules as R
import tools as T


class Pending(unittest.TestCase):
    def test_status_reask(self):
        for u in ["Hast du's schon gemacht?", "Hast du den Termin eingetragen?", "Hast du schon gemacht?", "Hast du den Termin schon eingetragen?", "Ist das schon eingetragen?"]:
            self.assertEqual(R.pending_decide("confirm", u), "status", u)
            self.assertEqual(R.pending_decide("when", u), "status", u)
    def test_yes(self):
        for u in ["Ja", "ja bitte", "Bestätige, bitte eintragen.", "trag ein", "mach", "Mach das.", "Bestätige", "Klar, eintragen"]:
            self.assertEqual(R.pending_decide("confirm", u), "yes", u)
    def test_no(self):
        for u in ["nein", "Nein, lass das", "abbrechen", "lass es", "Nein danke, lieber nicht", "nicht eintragen"]:
            self.assertEqual(R.pending_decide("confirm", u), "no", u)
    def test_new_request(self):
        for u in ["Wie wird das Wetter morgen?", "Was habe ich heute für Termine?", "Trag morgen um halb drei Zahnarzt in den Kalender ein und lade Max ein"]:
            self.assertIsNone(R.pending_decide("confirm", u), u)
    def test_when_stage(self):
        self.assertEqual(R.pending_decide("when", "nein, lass"), "no")
        self.assertIsNone(R.pending_decide("when", "um 18 Uhr"))
        self.assertIsNone(R.pending_decide("when", "am Freitag"))


class Misc(unittest.TestCase):
    def test_false_start(self):
        for u in ["oder", "Sport.", "ja", "Nein.", "", "Kizune."]: self.assertTrue(R.false_start(u), u)
        for u in ["Wie spät ist es?", "Und was ist mit der Donau?", "Lies vor", "Wetter in Basel", "Sag mal"]: self.assertFalse(R.false_start(u), u)
    def test_ignore(self):
        self.assertTrue(R.not_addressed("Ignoriert das bitte, Kizune, das ist nicht an dich gerichtet."))
        self.assertTrue(R.not_addressed("Ich rede nicht mit dir, nicht mit dir"))
        self.assertFalse(R.not_addressed("Wie wird das Wetter?"))
    def test_wake(self):
        for u in ["Kitsune, bist du da?", "Kizude", "Hey Kizune", "Wie spät ist es?"]: self.assertTrue(R.wakes(u), u)
        for u in ["Ja, genau das meine ich", "Also ich glaube nicht"]: self.assertFalse(R.wakes(u), u)
    def test_echo(self):
        self.assertTrue(R.echo("Hallo Kizune, wie geht es dir heute?", "Hallo nochmal, Kizune. Wie geht es dir heute?"))
        self.assertTrue(R.echo("Alles gut, Kizude.", "Alles gut, Kizude."))
        self.assertFalse(R.echo("Hallo!", "Hallo, hörst du mich?"))
        self.assertFalse(R.echo("Gern geschehen.", "Danke, das war's."))
    def test_stop(self):
        self.assertTrue(R.stop_word("Stopp!")); self.assertFalse(R.stop_word("Stopp, ich will das Wetter wissen"))


class Wake(unittest.TestCase):
    def test_name_variants(self):
        for u in ["Kitsune, bist du da?", "Kizune", "Kizude", "Kitsume, wie spät ist es", "Kitsuni", "Kissune?", "Isune, bist du noch da?", "Zunge, wie spät ist es", "Hey Kit sune", "Haben wir nochmal die Zune. Wie geht es dir", "Hallo nochmal, Kizune. Wie geht es dir heute?"]:
            self.assertTrue(R.NAME.search(u), u)
    def test_no_name(self):
        for u in ["Die Kommune hat beschlossen", "Die Sonne scheint", "Wie spät ist es?", "Kommissar Weber steht im Flur", "Seine Kollegin antwortet.", "Ja, genau", "Fortuna Düsseldorf gewinnt", "Das Gewitter kommt", "Die Tribune ist voll"]:
            self.assertFalse(R.NAME.search(u), u)
    def test_strip(self):
        self.assertEqual(R.strip_name("Hey Kitsune, wie spät ist es?"), "Wie spät ist es?")
        self.assertTrue(R.only_name("Kitsune?")); self.assertTrue(R.only_name("Hallo Kizune")); self.assertFalse(R.only_name("Kitsune, hallo du"))
    def test_end(self):
        for u in ["Kitsune, Pause", "Kitsune Schluss", "Danke, das war's.", "Kizune, das wäre alles"]: self.assertTrue(R.ends_conversation(u), u)
        for u in ["Wie ist die Pause beim Fußball geregelt?", "Danke", "Wie wird das Wetter?"]: self.assertFalse(R.ends_conversation(u), u)
    def test_ignore_variants(self):
        for u in ["Das ist nicht an die Gerichte", "ignorier das", "Ich red nicht mit dir", "Hör nicht zu", "Das war nicht an dich gerichtet"]: self.assertTrue(R.not_addressed(u), u)
    def test_gate_hard(self):
        g = lambda q, **k: R.gate_hard(q, **{"in_window": False, "pending": False, "ignoring": False, **k})
        self.assertEqual(g("Kitsune wie spät"), "pass"); self.assertEqual(g("Oder so"), "drop"); self.assertEqual(g("Sport."), "drop")
        self.assertEqual(g("Mehrere hundert Kilometer Strecke"), None); self.assertEqual(g("Vielen Dank."), "drop")
        self.assertEqual(g("Und die Donau?", in_window=True), "pass"); self.assertEqual(g("hallo", ignoring=True), "pass")
        self.assertEqual(g("Stopp"), "pass"); self.assertEqual(g("Ja", pending=True), "pass")
    def test_false_start_1word(self):
        self.assertTrue(R.false_start("Wetter?")); self.assertFalse(R.false_start("Wie spät?"))


class Endpoint(unittest.TestCase):
    def test_incomplete(self):
        for u in ["Trag morgen um halb drei", "Sport oder", "Setz einen Timer auf zehn", "Ich fahre am", "Wetter in"]: self.assertTrue(R.incomplete(u), u)
        for u in ["Wie viele Einwohner hat Hamburg?", "Wie spät ist es?", "Trag morgen um halb drei Zahnarzt ein", "Termin um 9 Uhr"]: self.assertFalse(R.incomplete(u), u)
    def test_title(self):
        for u in ["Trag morgen um halb drei", "Termin um 9 Uhr", "", "Mach mir einen Termin"]: self.assertTrue(R.title_missing(u), u)
        for u in ["Zahnarzt", "Trag morgen um halb drei Zahnarzt ein", "Meditation"]: self.assertFalse(R.title_missing(u), u)


class Templates(unittest.TestCase):
    JS = {"ort": "Basel", "jetzt": {"temp": 7, "wetter": "leicht bewölkt", "wind": 6}, "tage": [
        {"wochentag": "Sonntag", "min": 3, "max": 12, "wetter": "leichter Regen", "regen_prob": 60, "wind_max": 15},
        {"wochentag": "Montag", "min": -1, "max": 10, "wetter": "Nebel", "regen_prob": 0, "wind_max": 8}]}
    def test_wetter(self):
        s = T.say_wetter(self.JS, "")
        self.assertIn("In Basel", s); self.assertIn("drei bis zwölf Grad", s); self.assertIn("minus eins bis zehn Grad", s); self.assertIn("fünfzehn Kilometer", s)
        self.assertTrue(T.say_wetter(self.JS, "morgen").startswith("In Basel morgen"))
        self.assertIsNone(T.say_wetter(self.JS, "Freitag"))
    def test_nw(self):
        self.assertEqual([T.nw(n) for n in (0, 1, 21, 30, 100, 345)], ["null", "ein", "einundzwanzig", "dreißig", "einhundert", "dreihundertfünfundvierzig"])
        self.assertEqual(T.nw(1, True), "eins")
    def test_kal(self):
        ev = [{"title": "Zahnarzt 🦷", "start": "2026-11-02T14:30:00+01:00", "end": "2026-11-02T15:00:00+01:00"},
              {"title": "Feiertag (regional)", "start": "2026-11-02T00:00:00+01:00", "allDay": True},
              {"title": "Feiertag", "start": "2026-11-02T00:00:00+01:00", "allDay": True}]
        s = T.say_kal(ev, "today")
        self.assertIn("zwei Termine", s); self.assertIn("um vierzehn Uhr dreißig Zahnarzt", s); self.assertEqual(s.count("Feiertag"), 1)
        self.assertEqual(T.say_kal([], "today"), "Heute steht nichts im Kalender.")
    def test_gpu(self):
        s = T.say_gpu("GPU frei\nRAM frei ~40.5 GB, offen reserviert ~0 GB, Swap 12.3 GB")
        self.assertIn("GPU ist frei", s); self.assertIn("vierzig", s)

class Sep(unittest.TestCase):
    def test_particle(self):
        for q in ("Was steht heute an?", "Mach das Licht an.", "Rufe Anna an?"): self.assertFalse(R.incomplete(q), q)
        for q in ("Wie wird das Wetter morgen in", "Ich fahre morgen an", "Das Meeting beginnt um"): self.assertTrue(R.incomplete(q), q)


class FastRoute(unittest.TestCase):
    def test_hit(self):
        for q, n in (("Was steht heute an?", "kal_heute"), ("Welche Termine habe ich heute?", "kal_heute"), ("Was läuft gerade auf der GPU?", "gpu_status"), ("Wie ist die GPU ausgelastet?", "gpu_status"),
                     ("Wie viele Einwohner hat Hamburg?", "fakt"), ("Was ist die Hauptstadt von Bulgarien?", "fakt")):
            self.assertEqual((R.fast_route(q) or (None,))[0], n, q)
    def test_name_stripped(self):
        self.assertEqual(R.fast_route("Kitsune, was steht heute an?"), ("kal_heute", {}))
        self.assertIsNone(R.fast_route("Kitsune, bist du da?"))
    def test_miss(self):
        for q in ("Was steht morgen an?", "Trag heute einen Termin ein", "Wie wird das Wetter heute?", "Wie viele Einwohner hat Hamburg und wie ist das Wetter dort bei Regen?", "Was steht heute in der Zeitung?", "Wann ist die GPU Konferenz?"):
            self.assertIsNone(R.fast_route(q), q)

class SelfEcho(unittest.TestCase):
    A = "Mir geht es gut, danke der Nachfrage. Ich bin bereit, dir bei deinen Fragen zu helfen."
    def test_echo(self):
        for q in ("Ich.", "Ich bin bereit", "Bei deinen Fragen zu helfen", "Danke der Nachfrage."): self.assertTrue(R.self_echo(q, self.A), q)
    def test_user(self):
        for q in ("Stopp.", "Stopp, und warum ist er bunt?", "Kitsune, stopp", "Wie spät ist es?", "Warum ist der Himmel blau", "Hast du was gesagt?"): self.assertFalse(R.self_echo(q, self.A + " Der Himmel ist blau, weil Licht gestreut wird."), q)


class WebRoute(unittest.TestCase):
    def test_hit(self):
        for q, a in (("ist du im Web für mich nachschauen, ist Kim", "Kim"), ("Suche im Internet bitte nach Kimig.", "Kimig"), ("Such mal im Netz nach Mozart", "Mozart"), ("Google mal die Öffnungszeiten vom Zoo", "die Öffnungszeiten vom Zoo")):
            self.assertEqual(R.fast_route(q), ("netz_suche", {"frage": a}), q)
        self.assertEqual(R.fast_route("Suche im Internet")[0], "netz_ask"); self.assertEqual(R.fast_route("Kitsune, such im Internet nach Chemie."), ("netz_suche", {"frage": "Chemie"}))
    def test_miss(self):
        for q in ("Wie wird das Wetter online", "Was steht heute im Kalender, such online", "Wer war Goethe?"): self.assertNotEqual((R.fast_route(q) or ("",))[0], "netz_suche", q)
    def test_cont(self):
        self.assertTrue(R.is_continuation("und zwar KimiC")); self.assertFalse(R.is_continuation("Also ich glaube das stimmt nicht")); self.assertEqual(R.strip_cont("Und zwar KimiC."), "KimiC")


class Wx(unittest.TestCase):
    def test_regnet(self): self.assertTrue(R.WX.search("Regnet es heute in Hamburg?"))


class ChatDemote(unittest.TestCase):
    def test_demote(self):
        D = R.chat_demote
        for n, q in (("kal_heute", "Ich habe heute Banitsa gebacken und sie ist leider verbrannt."), ("kal_heute", "Du bist ein dummer Assistent."), ("wissen", "Hast du eine Lieblingsfarbe?"),
                     ("kal_eintragen", "Mein Bruder hat morgen Geburtstag."), ("wiki_suche", "Ich war gestern in Plowdiw, das war richtig schön."), ("wissen", "Ich bin heute irgendwie müde."), ("wetter", "Ich glaube, es regnet bald, mir tut der Kopf weh.")):
            self.assertEqual(D(n, q), "plaudern", q)
        self.assertEqual(D("kal_heute", "Welches Datum haben wir heute?"), "wissen"); self.assertEqual(D("kal_heute", "Wie geht es dir heute?"), "plaudern")
    def test_keep(self):
        D = R.chat_demote
        for n, q in (("kal_heute", "Was steht heute an?"), ("kal_woche", "Was habe ich nächste Woche vor?"), ("kal_eintragen", "Trag morgen um halb drei Zahnarzt ein."), ("kal_eintragen", "Morgen um zehn Zahnarzt"),
                     ("wiki_suche", "Wer war Napoleon?"), ("wissen", "Kannst du mir erklären, wie eine Wärmepumpe funktioniert?"), ("wiki_suche", "Zar Simeon"), ("kal_heute", "Was geht heute an?"), ("wetter", "Regnet es heute in Hamburg?"), ("wetter", "Wie wird das Wetter morgen?"), ("wetter", "Heute Regen in Hamburg")):
            self.assertIsNone(D(n, q), q)
        self.assertIsNone(R.chat_demote("kal_heute", "Ich habe heute Banitsa gebacken", pending=True))


class Grounded(unittest.TestCase):
    def test(self):
        self.assertFalse(R.grounded("[Ortsinformationen werden benötigt]", "Wie ist das Wetter heute?"))
        self.assertTrue(R.grounded("Basel", "Wie wird das Wetter morgen in Basel?")); self.assertTrue(R.grounded("Sunny Beach", "Wetter in Sunny Beach?"))
        self.assertIsNotNone(R.WX.search("Regnet es?")); self.assertIsNone(R.WX.search("Ist das deine Maximalsgeschwindigkeit?"))
        self.assertEqual(R.chat_demote("kal_woche", "Gut, gut. Kannst du auch schneller sprechen oder ist das deine Maximalsgeschwindigkeit?"), "plaudern")


class Tempo(unittest.TestCase):
    def test(self):
        for q, w in [("Sprich bitte schneller.", "faster"), ("Kannst du langsamer reden?", "slower"), ("Nicht so schnell!", "slower"), ("Sprich wieder normal.", "normal"),
                     ("Schneller sprechen bitte.", "faster"), ("Gut, gut. Kannst du auch schneller sprechen oder ist das deine Maximalgeschwindigkeit?", None)]:
            self.assertEqual(R.tempo_cmd(q), w, q)
        self.assertIsNone(R.tempo_cmd("Wie schnell ist ein Gepard?")); self.assertIsNone(R.tempo_cmd("Er spricht langsamer als ich."))


class BargeStop(unittest.TestCase):
    def test_stop(self):
        for q in ["Okay, nein, nein, nein, stopp.", "Warte mal.", "Hör auf!", "Halt, halt.", "Nein, nein.", "Sei still bitte."]: self.assertTrue(R.barge_stop(q), q)
        self.assertTrue(R.barge_contq("Und warum ist der?")); self.assertFalse(R.barge_contq("Und was machen wir heute Abend, Schatz?")); self.assertFalse(R.barge_contq("Und nun zu den Sportnachrichten."))
        self.assertTrue(R.barge_no("Okay, nein.")); self.assertFalse(R.barge_no("Nein, ich meine eigentlich morgen früh."))
        for q in ["Nein, ich meine morgen.", "Wie wird das Wetter?", "Kannst du auch schneller sprechen oder ist das deine Maximalgeschwindigkeit?", ""]: self.assertFalse(R.barge_stop(q), q)


class Ellipsis(unittest.TestCase):
    def test_slots(self):
        for q, want in [("Bitte in Berlin.", {"ort": "Berlin"}), ("und in Basel?", {"ort": "Basel"}), ("und morgen?", {"wann": "morgen"}),
                        ("Und übermorgen in Sofia?", {"ort": "Sofia", "wann": "übermorgen"}), ("In Bad Homburg bitte.", {"ort": "Bad Homburg"}),
                        ("Was ist mit Berlin?", {"ort": "Berlin"}), ("wie sieht es in Wien aus?", {"ort": "Wien"}), ("Ja, für Plowdiw.", {"ort": "Plowdiw"}), ("und am Samstag?", {"wann": "samstag"})]:
            self.assertEqual(R.ellipsis(q), want, q)
    def test_none(self):
        for q in ["und dann?", "und in der Küche?", "und mit mir?", "Danke.", "ja", "Okay", "Wie wird das Wetter morgen in Basel?", "Trag morgen um halb drei Zahnarzt ein."]:
            self.assertIsNone(R.ellipsis(q), q)


class Lex(unittest.TestCase):
    def test_spell(self):
        import lexicon as L
        self.assertEqual(L.spell("und zwar K I M I C"), "und zwar KIMIC"); self.assertEqual(L.spell("ich habe a b nicht"), "ich habe a b nicht")
        self.assertEqual(L.apply("und zwar K I M I C"), "und zwar KimiC"); self.assertEqual(L.apply("Suche nach Kimig."), "Suche nach KimiC.")


if __name__ == "__main__": unittest.main()
