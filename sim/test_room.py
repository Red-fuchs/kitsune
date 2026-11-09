import unittest, numpy as np
import room as R


class T(unittest.TestCase):
    def test_rt60(self):
        for name in ("buero", "kueche", "hotel"):
            p = R.PRESETS[name]; L = np.array(p.dim); m = L * [.3, .5, 0] + [0, 0, .95]
            h = R.rir(L, m + [0.9, 0.1, 0.2], m, p.rt60)
            est = R.rt60_est(h); self.assertLess(abs(est - p.rt60) / p.rt60, 0.4, (name, est))

    def test_direct_first(self):
        L = np.array([4, 3.5, 2.6]); m = np.array([1.2, 1.7, .95]); s = m + [0.9, 0, 0.2]
        h = R.rir(L, s, m, 0.35); d = np.linalg.norm(s - m)
        self.assertLess(abs(np.argmax(np.abs(h)) - d / R.C * R.SR), 2)

    def test_energy_ratio(self):  # direct/reverb: closer source -> more direct energy
        L = np.array([4, 3.5, 2.6]); m = np.array([1.2, 1.7, .95])
        dr = lambda d: (lambda h: (h[:40] ** 2).sum() / (h[40:] ** 2).sum())(R.rir(L, m + [d, 0, 0], m, 0.7))
        self.assertGreater(dr(0.3), dr(2.0))

    def test_speaker_path(self):
        sp = R.SpeakerPath(R.Room("buero")); x = np.random.default_rng(0).standard_normal(24000) * 0.1
        y = np.concatenate([sp.process(x[i:i + 1024]) for i in range(0, 24000, 1024)])
        self.assertAlmostEqual(len(y) / 16000, 1.0, delta=0.05)


if __name__ == "__main__": unittest.main()
