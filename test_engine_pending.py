"""Pending-confirmation state machine without loading models: run with ~/.venvs/mlxvlm/bin/python -I -m unittest discover -s . -p 'test_engine_pending.py'"""
import unittest, os, json, pathlib, tempfile
from datetime import datetime
os.environ["KITSUNE_DRYRUN"] = "1"
import engine as E, dates as D

NOW = datetime(2026, 11, 1, 12, 0)


def mk():
    k = E.Kitsune.__new__(E.Kitsune); k.pending, k.dry, k.last_items, k.prev_user = None, True, [], ""; return k


class T(unittest.TestCase):
    def start(self, k, utter="Trag morgen um halb drei Zahnarzt ein."):
        say, _ = k.exec_tool("kal_eintragen", {"titel": "Zahnarzt", "wann": "morgen um halb drei"}, utter, NOW)
        self.assertEqual(k.pending["stage"], "confirm"); return say
    def test_status_reasks_same_question(self):
        k = mk(); ask = self.start(k)
        for u in ["Hast du's schon gemacht?", "Hast du den Termin eingetragen?"]:
            p, k.pending = k.pending, None
            r = k.handle_pending(p, u, NOW)
            self.assertTrue(r.endswith(ask), r); self.assertEqual(k.pending["stage"], "confirm"); self.assertEqual(k.pending["ask"], ask)
    def test_yes_no(self):
        for u, want in [("ja", "Probelauf"), ("bestätige", "Probelauf"), ("trag ein", "Probelauf"), ("nein", "lasse ich es"), ("abbrechen", "lasse ich es"), ("lass", "lasse ich es")]:
            k = mk(); self.start(k); p, k.pending = k.pending, None
            self.assertIn(want, k.handle_pending(p, u, NOW), u)
    def test_new_request_passes_through(self):
        k = mk(); self.start(k); p, k.pending = k.pending, None
        self.assertIsNone(k.handle_pending(p, "Wie wird das Wetter morgen in Basel?", NOW))
    def test_when_stage_status(self):
        k = mk(); k.exec_tool("kal_eintragen", {"titel": "Yoga", "wann": "Freitag"}, "Trag am Freitag Yoga ein.", NOW)
        self.assertEqual(k.pending["stage"], "when"); p, k.pending = k.pending, None
        r = k.handle_pending(p, "Hast du schon gemacht?", NOW); self.assertTrue(r.startswith("Noch nicht")); self.assertEqual(k.pending["stage"], "when")
    def test_what_stage(self):
        k = mk(); say, _ = k.exec_tool("kal_eintragen", {"titel": "Trag morgen um halb drei", "wann": "morgen um halb drei"}, "Trag morgen um halb drei", NOW)
        self.assertEqual(say, "Was soll ich eintragen?"); self.assertEqual(k.pending["stage"], "what"); p, k.pending = k.pending, None
        r = k.handle_pending(p, "Zahnarzt ein.", NOW); self.assertIn("Zahnarzt", r); self.assertEqual(k.pending["stage"], "confirm")
    def test_wetter_ort_followup(self):
        fx = pathlib.Path(tempfile.mkdtemp()) / "wx.json"
        fx.write_text(json.dumps({"ort": "Basel", "jetzt": {"temp": 7, "wetter": "klar", "wind": 3}, "tage": [{"wochentag": "Sonntag", "min": 3, "max": 12, "wetter": "klar", "regen_prob": 0, "wind_max": 9}]}))
        os.environ["KITSUNE_WX_FIXTURE"] = str(fx)
        try:
            k = mk(); say, _ = k.exec_tool("wetter", {"ort": "", "wann": ""}, "Wie ist das Wetter?", NOW)
            self.assertEqual(say, "Für welchen Ort?"); p, k.pending = k.pending, None
            self.assertIn("In Basel", k.handle_pending(p, "In Basel.", NOW))
        finally: del os.environ["KITSUNE_WX_FIXTURE"]

class Place(unittest.TestCase):
    def test_place_correction(self):
        fx = pathlib.Path(tempfile.mkdtemp()) / "wx.json"
        fx.write_text(json.dumps({"ort": "Plowdiw", "jetzt": {"temp": 9, "wetter": "klar", "wind": 3}, "tage": [{"wochentag": "Sonntag", "min": 3, "max": 12, "wetter": "klar", "regen_prob": 0, "wind_max": 9}]}))
        os.environ["KITSUNE_WX_FIXTURE"] = str(fx)
        try:
            k = mk(); say, _ = k.exec_tool("wetter", {"ort": "Blau tief", "wann": ""}, "Wetter in Blau tief", NOW)
            self.assertEqual(say, "Meinst du Plowdiw?"); p, k.pending = k.pending, None
            self.assertIn("In Plowdiw", k.handle_pending(p, "Ja", NOW)); self.assertEqual(k.last_place, "Plowdiw")
            k = mk(); say, _ = k.exec_tool("wetter", {"ort": "Plovdiv", "wann": ""}, "Wetter in Plovdiv", NOW); self.assertIn("In Plowdiw", say)
        finally: del os.environ["KITSUNE_WX_FIXTURE"]

if __name__ == "__main__": unittest.main()


class Bridge(unittest.TestCase):
    """filler chain: another short filler only while the playhead is about to run dry and no answer audio was fed yet."""
    def mkb(self):
        import threading, numpy as np
        k = mk(); k.fillers = {x: np.zeros(2400, np.float32) for x in E.BRIDGE}; k.cancel = threading.Event()
        k._fl, k._pe, k.ans_started, k._bth, k.watch, k.t_end0 = threading.Lock(), 0.0, False, None, None, 0.0
        class S(E.WavSink): skip = 0
        return k, S()
    def test_chain_then_stop(self):
        import time
        k, s = self.mkb(); m = {}; k._pe = time.monotonic() + 0.1  # a filler is playing, ends in 0.1 s
        E.CHAIN = 2; k.bridge_start(s, m, time.perf_counter()); time.sleep(0.6); k.bridge_end()
        self.assertEqual(m.get("bridges"), 2)  # capped at CHAIN
    def test_answer_stops_chain(self):
        import time
        k, s = self.mkb(); m = {}; k._pe = time.monotonic() + 0.3; E.CHAIN = 2
        k.bridge_start(s, m, time.perf_counter()); k.ans_started = True; time.sleep(0.6); k.bridge_end()
        self.assertEqual(m.get("bridges", 0), 0)
    def test_off(self):
        k, s = self.mkb(); E.CHAIN = 0; k.bridge_start(s, {}, 0); self.assertIsNone(k._bth); E.CHAIN = 2
