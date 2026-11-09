"""Streaming WSOLA time stretch (pitch preserved) for 24 kHz TTS chunks. rate > 1 = faster."""
import numpy as np


class Stretch:
    def __init__(self, rate, n=960, tol=240):
        self.r, self.n, self.hs, self.tol = rate, n, n // 2, tol
        self.ha = self.hs * rate
        self.w = np.hanning(n + 1)[:n].astype(np.float32)  # periodic Hann: overlap-add at hop n/2 sums to 1
        self.x, self.x0, self.k, self.prev = np.zeros(0, np.float32), 0, 0, None
        self.acc = np.zeros(n, np.float32)

    def _seg(self, a, m):
        i = a - self.x0; s = self.x[max(i, 0):i + m]
        return np.pad(s, (max(-i, 0), m - len(s) - max(-i, 0))) if len(s) < m else s

    def push(self, pcm, final=False):
        if self.r == 1: return np.asarray(pcm, np.float32)
        self.x = np.concatenate([self.x, np.asarray(pcm, np.float32)]); out = []
        while True:
            a = int(round(self.k * self.ha)); end = self.x0 + len(self.x)
            lo, hi = (a, a) if self.prev is None else (max(a - self.tol, self.x0), a + self.tol)
            need = max(hi + self.n, (self.prev + self.hs + self.n) if self.prev is not None else 0)
            if need > end and (not final or a >= end): break
            if self.prev is None: best = a
            else:  # the frame that best continues the previous one (waveform similarity)
                c = np.correlate(self._seg(lo, hi - lo + self.n), self._seg(self.prev + self.hs, self.n), "valid"); best = lo + int(np.argmax(c))
            self.acc += self._seg(best, self.n) * self.w
            out.append(self.acc[:self.hs].copy()); self.acc = np.concatenate([self.acc[self.hs:], np.zeros(self.hs, np.float32)])
            self.prev, self.k = best, self.k + 1
            cut = min(self.prev, int(round(self.k * self.ha)) - self.tol) - self.x0
            if cut > 0: self.x, self.x0 = self.x[cut:], self.x0 + cut
        if final: out.append(self.acc[:self.n - self.hs].copy()); self.acc[:] = 0
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def stretch(a, rate):
    s = Stretch(rate); return np.concatenate([s.push(a), s.push(np.zeros(0, np.float32), final=True)])
