"""Opt-in Apple Voice Processing (AEC) audio path (`kitsune --aec` / KITSUNE_AEC=1): a Swift helper (aec/kitsune-aec) plays our PCM AND captures the mic through ONE voice-processing unit,
so the mic stream is echo-cancelled against exactly what we play. This module = helper process + sounddevice shim (InputStream) + AecSink (SpeakerSink interface).
Any failure (no binary, no permission, helper dies) -> fallback to the normal sounddevice path (new streams/sinks are real ones again). Protocol: see aec/main.swift."""
import os, sys, json, time, struct, queue, threading, subprocess, pathlib
import numpy as np

EXE = pathlib.Path(__file__).parent / "aec" / "kitsune-aec"
SR_OUT, SR_IN = 24000, 16000
_inst = {"aec": None, "real": None}


class Aec:
    def __init__(self, exe=None, args=()):
        self.p = subprocess.Popen([str(exe or EXE), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(pathlib.Path(__file__).parent / "logs" / "aec.log", "ab") if (pathlib.Path(__file__).parent / "logs").exists() else subprocess.DEVNULL, bufsize=0)
        self.alive, self.hello, self.err = True, None, None
        self.consumed, self.queued, self.under = 0, 0, 0
        self.t_tel = 0.0
        self.subs, self.lock, self.wlock = [], threading.Lock(), threading.Lock()
        self.ev = threading.Event()
        threading.Thread(target=self._rd, daemon=True).start()

    # ---- helper -> python
    def _rn(self, n):
        b = b""
        while len(b) < n:
            c = self.p.stdout.read(n - len(b))
            if not c: raise EOFError
            b += c
        return b

    def _rd(self):
        try:
            while True:
                h = self._rn(5); t, n = h[0], struct.unpack("<I", h[1:])[0]; pl = self._rn(n) if n else b""
                if t == 0x4d:
                    x = np.frombuffer(pl, np.float32).copy()
                    with self.lock:
                        for s in self.subs: s.put(x)
                elif t == 0x54 and n >= 20:
                    self.consumed, self.queued, self.under = struct.unpack("<QQI", pl[:20]); self.t_tel = time.monotonic()
                elif t == 0x48: self.hello = json.loads(pl); self.ev.set()
                elif t == 0x45: self.err = pl.decode(errors="replace"); self.ev.set()
        except Exception: pass
        finally: self.alive = False; self.ev.set()

    def wait_ready(self, timeout=6.0):
        self.ev.wait(timeout)
        return self.alive and self.hello is not None

    # ---- python -> helper
    def send(self, t, payload=b""):
        if not self.alive: return False
        try:
            with self.wlock: self.p.stdin.write(bytes([t]) + struct.pack("<I", len(payload)) + payload)
            return True
        except Exception: self.alive = False; return False

    def close(self):
        self.send(0x51)
        try: self.p.wait(1.0)
        except Exception: self.p.kill()
        self.alive = False
        for f in (self.p.stdin, self.p.stdout):
            try: f.close()
            except Exception: pass


class AecInputStream:
    """sounddevice.InputStream subset used by listen()/Watch/BargeIn: start/stop/close/read(n) -> (x[n,1], overflow). Starts with fresh audio at start()."""
    def __init__(self, aec, blocksize=1600, **kw):
        self.a, self.bs, self.q, self.buf, self.live = aec, blocksize, queue.Queue(maxsize=400), np.zeros(0, np.float32), False
    def start(self):
        with self.a.lock: self.a.subs.append(self.q)
        self.live = True
    def stop(self):
        with self.a.lock:
            if self.q in self.a.subs: self.a.subs.remove(self.q)
        self.live = False
    close = stop
    def read(self, n):
        t0 = time.monotonic()
        while len(self.buf) < n:
            try: self.buf = np.concatenate([self.buf, self.q.get(timeout=0.2)])
            except queue.Empty:
                if not self.a.alive or time.monotonic() - t0 > 2.0:   # helper died: keep pacing with silence; the next stream is a real one again
                    time.sleep(n / SR_IN); return np.zeros((n, 1), np.float32), False
        x, self.buf = self.buf[:n], self.buf[n:]
        return x[:, None].copy(), False


class SDShim:
    """stands in for `sounddevice` while AEC is up: InputStream -> helper mic, everything else (query_devices ...) -> the real module."""
    def __init__(self, real): self._real = real
    def InputStream(self, *a, **kw):
        ae = _inst["aec"]
        if ae is not None and ae.alive: return AecInputStream(ae, kw.get("blocksize") or 1600)
        return self._real.InputStream(*a, **kw)   # fallback
    def __getattr__(self, k): return getattr(self._real, k)


def start(exe=None, args=(), timeout=6.0, log=print):
    """-> SDShim (installed as sys.modules['sounddevice']) or None (reason logged). Never raises."""
    try:
        import sounddevice as real
        if not pathlib.Path(exe or EXE).exists(): log(f"aec: {EXE} fehlt (cd aec && ./build.sh)"); return None
        a = Aec(exe, args)
        if not a.wait_ready(timeout): log("aec: Helfer nicht bereit: " + (a.err or "Timeout / Mikrofon-Berechtigung?")); a.close(); return None
        _inst["aec"], _inst["real"] = a, real
        shim = SDShim(real); sys.modules["sounddevice"] = shim
        log(f"aec: Voice Processing aktiv {a.hello}"); return shim
    except Exception as e:
        log(f"aec: aus ({e})"); return None


def stop():
    a = _inst["aec"]
    if a: a.close()
    _inst["aec"] = None
    if _inst["real"] is not None: sys.modules["sounddevice"] = _inst["real"]


def ok(): return _inst["aec"] is not None and _inst["aec"].alive


class _Buf:  # len() = samples fed but not yet played (BargeIn only asks len(sink.buf))
    def __init__(self, f): self.f = f
    def __len__(self): return self.f()


def make_sink_cls(base):
    """base = engine.WavSink"""
    class AecSink(base):
        """SpeakerSink interface on the helper. No OutputStream. aec=True lets BargeIn pick the AEC profile."""
        aec = True
        def __init__(self, pre=None):
            super().__init__()
            a = _inst["aec"]; self.a = a; pre = 0.25 if pre is None else pre
            self.playing, self.under, self.xruns, self.done, self.paused, self.t_play, self.stopped = False, 0, 0, False, False, None, False
            self.fed, self.out, self.skip, self.log = 0, 0, 0, []
            self.buf = _Buf(lambda: 0 if self.stopped else max(0, self.fed - self.out))
            self.st = type("S", (), {"latency": 0.1})()
            a.send(0x42, struct.pack("<I", int(pre * SR_OUT)))
        def _sync(self):
            a = self.a
            if not self.stopped: self.out = min(int(a.consumed), self.fed)
            self.under = int(a.under)
            if self.out > 0 and not self.playing: self.playing = True; self.t_play = time.perf_counter() + 0.05
        def feed(self, pcm):
            if self.done: return
            pcm = np.asarray(pcm, np.float32); super().feed(pcm)
            self._sync(); t0 = time.monotonic() + max(0, self.fed - self.out) / SR_OUT + 0.05   # estimated time the chunk hits the speaker
            self.fed += len(pcm)
            self.log = [e for e in self.log if e[0] > time.monotonic() - 2.0] + [(t0 + i / SR_OUT, pcm[i:i + 1024]) for i in range(0, len(pcm), 1024)]
            self.a.send(0x50, pcm.tobytes())
        def played_between(self, t0, t1):
            blocks = [b for t, b in self.log if t0 <= t <= t1]
            return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
        def pause(self): self.paused = True; self.a.send(0x55)
        def resume(self): self.paused = False; self.a.send(0x52)
        def stop_now(self): self._sync(); self.done, self.paused, self.stopped = True, False, True; self.a.send(0x53)
        def frac_played(self): self._sync(); return max(0, self.out - self.skip) / (self.fed - self.skip) if self.fed > self.skip else 1.0
        def wait(self):
            self.done = True; self.a.send(0x46); t0 = time.monotonic()
            while not self.stopped and time.monotonic() - t0 < 120:
                self._sync()
                if self.out >= self.fed or not self.a.alive: break
                time.sleep(0.03)
            time.sleep(0.15)
    return AecSink
