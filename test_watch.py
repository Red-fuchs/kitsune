import unittest, time, numpy as np
from watch import Watch


class FakeSt:
    def __init__(self, pattern): self.p, self.i, self.closed = pattern, 0, False
    def read(self, n):
        time.sleep(0.01); a = self.p[self.i] if self.i < len(self.p) else 0.0; self.i += 1
        return (np.random.randn(n, 1).astype(np.float32) * a if a else np.zeros((n, 1), np.float32)), False
    def stop(self): pass
    def close(self): self.closed = True


class T(unittest.TestCase):
    def test_resume(self):
        st = FakeSt([0, 0, .1, .1, .1, .1, 0, 0, 0, 0, 0, 0]); w = Watch(st, 0.01, time.perf_counter(), max_s=5)
        self.assertTrue(w.wait_until(time.perf_counter() + 2)); x = w.collect()
        self.assertGreaterEqual(len(x), 9 * 1600); self.assertTrue(st.closed); self.assertGreater(w.extra_s, 0.8)
    def test_no_speech_window_ends(self):
        st = FakeSt([0] * 100); w = Watch(st, 0.01, time.perf_counter(), max_s=0.1)
        self.assertFalse(w.wait_until(time.perf_counter() + 0.3)); w.done.wait(1); self.assertTrue(st.closed); self.assertEqual(w.extra_s, 0)
    def test_close(self):
        st = FakeSt([0] * 1000); w = Watch(st, 0.01, time.perf_counter(), max_s=50); w.close(); w.done.wait(1); self.assertTrue(st.closed)


if __name__ == "__main__": unittest.main()
