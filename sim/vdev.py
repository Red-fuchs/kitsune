"""Virtual audio devices: a fake `sounddevice` (InputStream/OutputStream) backed by a World. Real-time: mic blocks are released on the wall clock.
mic(t) = user*RIR_u + kitsune_out*speaker_model*RIR_s + beds + self-noise.  Install with vdev.install(world)."""
import sys, time, threading, types
import numpy as np
from scipy import signal
import room as R, beds as B

SR = 16000
IN_LAT = 0.01          # input latency s
OUT_LAT = 0.10         # output latency s ("high" latency stream)


class SimEnd(Exception): pass


class World:
    def __init__(self, room, cap_s=600, seed=0):
        self.room = room if isinstance(room, R.Room) else R.Room(room); self.cap = cap_s * SR
        self.rng = np.random.default_rng(seed); self.lock = threading.RLock()
        self.user, self.echo, self.amb = (np.zeros(self.cap, np.float32) for _ in range(3))
        self.selfn = R.mic_hp(self.rng.standard_normal(self.cap) * R.MIC_NOISE).astype(np.float32)
        self.t0 = None; self.end_t = None
        self.says = []      # dict(text, start, end, voice)
        self.beds = []      # dict(name, start, end, db)
        self.spk = []       # (t_play_sim, rms, peak) per callback with audio
        self.xruns = 0; self.cb_late_ms = []
        self.kit = []       # kitsune playback segments [start, end] (sim s)
        self._open = None   # (start, last_sound)

    def start(self): self.t0 = time.monotonic()
    def now(self): return time.monotonic() - self.t0

    # ---- scheduled sources ----
    def add_user(self, x, t, level_db=0.0):
        """x: dry voice 16k. Placed at sim time t (>= already produced mic head is the caller's duty). Returns (start, end)."""
        a = x / (np.sqrt((x[np.abs(x) > 0.05 * np.abs(x).max()] ** 2).mean()) + 1e-9) * R.USER_RMS * 10 ** (level_db / 20)
        y = signal.fftconvolve(a, self.room.h_user)[:len(a) + 2000].astype(np.float32)
        i = int(t * SR)
        with self.lock: self.user[i:i + len(y)] += y[:self.cap - i]
        s, e = t, t + len(x) / SR; self.says.append(dict(start=s, end=e)); return s, e

    def add_bed(self, name, t, db=-20.0, dur=None, seed=0):
        x = B.make(name, 60, seed); n = int((dur or min(150, self.cap / SR - t - 1)) * SR); n = min(n, self.cap - int(t * SR))
        x = np.tile(x, n // len(x) + 1)[:n]
        if name in B.SPATIAL: x = signal.fftconvolve(x, self.room.h_noise)[:n]
        x = x / np.sqrt((x ** 2).mean()) * R.USER_RMS * 10 ** (db / 20)
        fade = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.2 * SR)); x = (x * fade).astype(np.float32)
        i = int(t * SR)
        with self.lock: self.amb[i:i + n] += x
        self.beds.append(dict(name=name, start=t, end=t + n / SR, db=db))

    # ---- speaker side ----
    def speaker_write(self, t_play, i, chunk16, raw24):
        """chunk16: echo contribution (already x echo gain, 16k) for audio hitting the speaker at sim time t_play."""
        if len(chunk16):
            y = signal.fftconvolve(chunk16, self.room.h_spk).astype(np.float32)
            with self.lock: self.echo[i:i + len(y)] += y[:max(0, self.cap - i)]
        pk = float(np.abs(raw24).max()) if len(raw24) else 0.0
        if pk > 1e-4:
            self.spk.append((t_play, float(np.sqrt((raw24 ** 2).mean())), pk))
            dur = len(raw24) / 24000
            if self._open is None: self._open = [t_play, t_play + dur]; self.kit.append(self._open)
            elif t_play - self._open[1] <= 0.7: self._open[1] = t_play + dur
            else: self._open = [t_play, t_play + dur]; self.kit.append(self._open)

    def mic(self, a, b):
        with self.lock: x = self.user[a:b] + self.echo[a:b] + self.amb[a:b] + self.selfn[a:b]
        return np.clip(x, -1, 1).astype(np.float32)

    def playing_at(self, t): return any(s <= t <= e for s, e in self.kit)


class InputStream:
    def __init__(self, samplerate=SR, channels=1, dtype="float32", blocksize=0, device=None, latency=None, **kw):
        assert samplerate == SR; self.w = WORLD[0]; self.cur = None
    def start(self): self.cur = int(self.w.now() * SR)
    def stop(self): pass
    def close(self): pass
    @property
    def read_available(self): return max(0, int((self.w.now() - IN_LAT) * SR) - self.cur)
    def read(self, n):
        w = self.w; need = (self.cur + n) / SR + IN_LAT
        while True:
            d = need - w.now()
            if (w.end_t is not None and w.now() > w.end_t) or self.cur + n > w.cap: raise SimEnd()
            if d <= 0: break
            time.sleep(min(d, 0.01))
        x = w.mic(self.cur, self.cur + n); self.cur += n
        return x[:, None], False


class _Status:
    output_underflow = False


class OutputStream:
    def __init__(self, samplerate=24000, channels=1, dtype="float32", callback=None, blocksize=1024, latency=None, **kw):
        assert samplerate == 24000 and callback; self.w = WORLD[0]; self.cb, self.bs = callback, blocksize
        self.pos = None; self.latency = OUT_LAT; self.stop_ev = threading.Event(); self.th = None; self.path = R.SpeakerPath(self.w.room)
    def start(self):
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()
    def _run(self):
        w, dt = self.w, self.bs / 24000; nxt = time.monotonic()
        while not self.stop_ev.is_set():
            d = nxt - time.monotonic()
            if d > 0: time.sleep(d)
            late = time.monotonic() - nxt
            st = _Status()
            if late > dt: st = type("S", (), {"output_underflow": True})(); w.xruns += 1; nxt = time.monotonic()
            w.cb_late_ms.append(late * 1000)
            out = np.zeros((self.bs, 1), np.float32); tcall = w.now()
            try: self.cb(out, self.bs, None, st)
            except Exception as e: print("sim output cb error", e, file=sys.stderr)
            x = out[:, 0].copy()
            y = self.path.process(x); iw = int(round((tcall + self.latency) * SR))
            if self.pos is None or abs(iw - self.pos) > 0.02 * SR: self.pos = iw  # keep chunks contiguous (no clicks from thread jitter)
            w.speaker_write(tcall + self.latency, self.pos, y, x); self.pos += len(y)
            nxt += dt
    def stop(self):
        self.stop_ev.set()
        if self.th and self.th is not threading.current_thread(): self.th.join(1)
    def close(self): self.stop()


WORLD = [None]


def install(world):
    WORLD[0] = world
    m = types.ModuleType("sounddevice"); m.InputStream, m.OutputStream = InputStream, OutputStream
    m.query_devices = lambda *a, **k: "virtual"
    sys.modules["sounddevice"] = m
