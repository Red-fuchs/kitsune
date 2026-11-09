"""AEC wiring against the real Swift helper in --null mode (protocol, no audio device, no mic): ~/.venvs/mlxvlm/bin/python -I -m unittest discover -s . -p 'test_aec.py'. Needs aec/kitsune-aec built (aec/build.sh)."""
import unittest, time, os, sys, pathlib
os.environ["KITSUNE_DRYRUN"] = "1"
import numpy as np
import aec
import engine as E

EXE = aec.EXE


@unittest.skipUnless(EXE.exists(), "aec/kitsune-aec not built")
class T(unittest.TestCase):
    def setUp(self):
        self.real = sys.modules.get("sounddevice")
        self.sh = aec.start(exe=EXE, args=("--null",), log=lambda s: None); self.assertIsNotNone(self.sh)
        self.S = aec.make_sink_cls(E.WavSink)
    def tearDown(self): aec.stop(); self.assertIs(sys.modules.get("sounddevice"), self.real)

    def test_mic_blocks(self):
        st = self.sh.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=1600); st.start()
        t = time.monotonic(); x, _ = st.read(1600); x2, _ = st.read(1600); st.close()
        self.assertEqual(x.shape, (1600, 1)); self.assertLess(time.monotonic() - t, 0.5); self.assertEqual(float(np.abs(x).max()), 0.0)

    def test_playback_pause_stop(self):
        s = self.S(pre=0.1); pcm = (0.1 * np.sin(np.arange(24000) / 7)).astype(np.float32)
        s.feed(pcm[:12000]); s.feed(pcm[12000:]); time.sleep(0.25); s._sync()
        self.assertTrue(s.playing); self.assertGreater(s.out, 0); self.assertLess(s.frac_played(), 0.6)
        s.pause(); time.sleep(0.1); o = s.frac_played(); time.sleep(0.3); self.assertAlmostEqual(o, s.frac_played(), delta=0.02)   # paused: nothing consumed
        self.assertGreater(len(s.played_between(time.monotonic() - 1, time.monotonic() + 2)), 0)
        s.resume(); time.sleep(0.2); self.assertGreater(s.frac_played(), o)
        s.stop_now(); f = s.frac_played(); self.assertEqual(len(s.buf), 0); t = time.monotonic(); s.wait(); self.assertLess(time.monotonic() - t, 1.0); self.assertAlmostEqual(f, s.frac_played(), delta=0.001)

    def test_wait_plays_everything(self):
        s = self.S(pre=0.5); s.feed(np.zeros(4800, np.float32)); t = time.monotonic(); s.wait()   # 0.2 s < pre-buffer: finish flag plays it anyway
        self.assertLess(time.monotonic() - t, 1.5); self.assertEqual(s.frac_played(), 1.0)

    def test_fallback_when_helper_dies(self):
        a = aec._inst["aec"]; a.p.kill(); a.p.wait(); time.sleep(0.3)
        self.assertFalse(aec.ok()); sentinel = object(); orig = E.SpeakerSink; E.SpeakerSink = lambda: sentinel
        try: self.assertIs(E.new_sink(), sentinel)   # plain audio again
        finally: E.SpeakerSink = orig


class Missing(unittest.TestCase):
    def test_no_binary(self):
        msgs = []; self.assertIsNone(aec.start(exe="/nonexistent/kitsune-aec", log=msgs.append)); self.assertTrue(msgs)
        self.assertIsNone(aec._inst["aec"])


if __name__ == "__main__": unittest.main()
