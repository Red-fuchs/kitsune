"""Standby/addressee gate state machine without models: ~/.venvs/mlxvlm/bin/python -I -m unittest discover -s . -p 'test_gate.py'"""
import unittest, time, os
os.environ["KITSUNE_DRYRUN"] = "1"
import engine as E


class FakeAddr:
    ms = 70
    def __init__(self, p): self.p = p
    def p_addr(self, q, ctx=()): return self.p
    def p_barge(self, q, speaking, ctx=()): return self.p
    def ask(self, q, ctx=(), tools=None, addr=True): return {"ms": self.ms, "p": self.p, **({"tool": "plaudern", "conf": 0.9} if tools else {})}


def mk(mode="standby", p=0.9):
    k = E.Kitsune.__new__(E.Kitsune)
    k.handsfree, k.mode, k.standby_s, k.t_act, k.t_kit, k.rejects, k.bg_until = True, mode, 45.0, time.monotonic(), -1e9, [], 0.0
    k.gate_ans, k.gate_hm, k.gate_bg, k.addr, k.hist, k.pending, k.ignoring = 0.8, 0.6, 0.9, FakeAddr(p), [], None, False
    return k


class T(unittest.TestCase):
    def test_standby(self):
        k = mk()
        self.assertEqual(k.gate("Was steht heute an?", None, False)[0], "drop")
        self.assertEqual(k.gate("Kizune?", None, False)[0], "wake"); self.assertEqual(k.mode, "active")
        k.t_kit = time.monotonic(); self.assertEqual(k.gate("Und morgen?", None, False)[0], "pass")
    def test_wake_with_request_and_end(self):
        k = mk(); v, i = k.gate("Kitsune, wie spät ist es?", None, False); self.assertEqual((v, i.get("woke")), ("pass", True))
        self.assertEqual(k.gate("Kitsune, Pause", None, False)[0], "end")
        k = mk(); self.assertEqual(k.gate("Kitsune Schluss", None, False)[0], "end"); self.assertEqual(k.mode, "standby")
    def test_timeout(self):
        k = mk("active"); k.t_act -= 100; self.assertEqual(k.gate("Wie spät ist es?", None, False)[0], "drop"); self.assertEqual(k.mode, "standby")
    def test_addressee(self):
        self.assertEqual(mk("active", .9).gate("Das Wetter bleibt heute meist wolkig im Norden", None, False)[0], "pass")
        self.assertEqual(mk("active", .3).gate("Das Wetter bleibt heute meist wolkig im Norden", None, False)[0], "drop")
        self.assertEqual(mk("active", .7).gate("Wie spät ist es bei dir gerade?", None, False)[0], "hm")
        self.assertEqual(mk("active", .7).gate("Das Wetter bleibt heute meist wolkig im Norden", None, False)[0], "drop")
    def test_background_profile(self):
        k = mk("active", .3)
        for _ in range(3): k.gate("Das Wetter bleibt heute meist wolkig im Norden", None, False)
        k.addr.p = .85; self.assertEqual(k.gate("Wie spät ist es bei dir gerade?", None, False)[0], "drop")  # strict for 60 s
        self.assertEqual(k.gate("Kitsune, wie spät ist es?", None, False)[0], "pass")
    def test_window(self):
        k = mk("active", .1); k.t_kit = time.monotonic() - 3
        self.assertEqual(k.gate("Und was ist mit der Donau?", None, False)[0], "pass")
        k.t_kit = time.monotonic() - 20; self.assertEqual(k.gate("Und was ist mit der Donau?", None, False)[0], "drop")

    def test_continuation_and_hm_after_answer(self):
        k = mk("active", .66); self.assertEqual(k.gate("und zwar K I M I C", None, False)[0], "drop")  # no recent answer -> as before
        k.t_ans = time.monotonic() - 12; self.assertEqual(k.gate("und zwar K I M I C", None, False)[1].get("why"), "continuation")
        k = mk("active", .8); k.t_ans = time.monotonic() - 12  # .8 * .8 long-declarative penalty = .64
        v, i = k.gate("Das Wetter bleibt heute meist wolkig im Norden", None, False); self.assertEqual((v, i.get("relaxed")), ("hm", True))  # middling p right after an exchange -> 'Hm?' (no window reopening)
        self.assertEqual(k.gate("Das Wetter bleibt heute meist wolkig im Norden", None, False)[0], "drop")  # but only once per 20 s
        k = mk("active", .8); k.t_ans = time.monotonic() - 12; self.assertEqual(k.gate("Das Wetter bleibt heute meist wolkig im Norden und es regnet weiter im Süden", None, False)[0], "drop")  # >8 words: no relaxed Hm?
        k = mk("active", .8); k.t_ans = time.monotonic() - 40; self.assertEqual(k.gate("Das Wetter bleibt heute meist wolkig im Norden", None, False)[0], "drop")


if __name__ == "__main__": unittest.main()
