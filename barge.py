"""Barge-in during playback, variant A: echo-aware detection without real AEC.
Per 50 ms mic block: 16 log bands (150-6000 Hz) vs expected echo = G[band] * weighted envelope of what was
played in the last 0.3 s (+ noise floor). G (room/speaker->mic gain) is learned while the user is silent and kept across turns.
A trigger first only PAUSES playback; the echo-free mic audio after it (<=1 s) goes to verify() (ASR + stop/question/addressee check, engine.barge_verify). TV/talk -> playback resumes and the band levels heard are
added to the expected level for 20 s (the TV is then no longer an onset); real -> hit + stop. verify=None -> old immediate stop.
User speech = >=40 % of bands above 4x expected echo in 2 of the last 3 blocks, or (fast path) mic power > 36x expected echo power for 2 blocks."""
import threading, time, os, json, pathlib, subprocess
import numpy as np

SR, BLK = 16000, 800  # 50 ms blocks
EDGES = np.geomspace(150, 6000, 17)
_bins = np.fft.rfftfreq(BLK, 1 / SR)
BANDS = [(_bins >= lo) & (_bins < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
WIN = np.hanning(BLK).astype(np.float32)


def bands(x):
    p = np.abs(np.fft.rfft(x[:BLK] * WIN)) ** 2
    return np.array([p[b].mean() for b in BANDS]) + 1e-12


G_MIN = float(os.environ.get("KITSUNE_G_MIN", "0.0003"))      # degenerate-calibration floor (headphones / unknown output)
G_MIN_SPK = float(os.environ.get("KITSUNE_G_MIN_SPK", "0.06"))  # built-in loudspeakers: measured G is 0.11-0.15 (log 2026-11-03); the first calibration once gave 0.017 (own word 'Ich' counted as user) -> never trust less than half of the usual value
_SPK = ("macbook", "lautsprecher", "speaker", "built-in", "eingebaut")
HOLD_S = 0.3   # no calibration / no trigger in the first 0.3 s of playback (echo has not reached the mic yet; blocks there give a too-low G)
CALIB_N = 12
G_FILE = pathlib.Path(__file__).parent / "cache" / "room_g.json"


class Room:  # learned once per process (room + volume rarely change); G prior persisted per mic|output|volume
    G = None
    G0 = None
    floor = None
    key = None
    persist = True        # sim sets False
    gmin = G_MIN
    aec = False
    calib_scale = 1.0     # test hook only: simulates a wrong first calibration
    _vol = (0.0, None)

    @classmethod
    def devkey(cls, mic):
        """mic|output device|output volume; volume is cached 20 s (osascript ~80 ms)."""
        t = time.monotonic()
        if t - cls._vol[0] > 20:
            try: v = subprocess.run(["osascript", "-e", "output volume of (get volume settings)"], capture_output=True, text=True, timeout=2).stdout.strip()
            except Exception: v = "?"
            try:
                import sounddevice as sd; out = sd.query_devices(kind="output")["name"]
            except Exception: out = "?"
            cls._vol = (t, f"{out}|{v}")
        cls.gmin = G_MIN_SPK if any(x in cls._vol[1].lower() for x in _SPK) else G_MIN
        return f"{mic}|{cls._vol[1]}" + ("|aec" if cls.aec else "")

    @classmethod
    def load_prior(cls, key):
        try: g = json.loads(G_FILE.read_text()).get(key)
        except Exception: g = None
        return np.maximum(np.array(g, dtype=float), cls.gmin) if g and len(g) == len(BANDS) else None

    @classmethod
    def save_prior(cls, key, G):
        try:
            d = json.loads(G_FILE.read_text()) if G_FILE.exists() else {}
            d[key] = [float(x) for x in G]; G_FILE.parent.mkdir(exist_ok=True); G_FILE.write_text(json.dumps(d))
        except Exception: pass


class BargeIn(threading.Thread):
    def __init__(self, sink, mic=None, ratio=4.0, frac=0.4, calib_s=0.6, silence_s=0.6, max_s=30, log=None, fast=6.0, verify=None):
        super().__init__(daemon=True)
        self.sink, self.mic, self.ratio, self.frac, self.calib_s = sink, mic, ratio, frac, calib_s
        self.silence_s, self.max_s, self.log, self.fast, self.verify = silence_s, max_s, log, fast, verify
        self.hit = threading.Event()      # barge-in detected → engine stops generating
        self.done = threading.Event()     # user utterance captured (or monitor ended)
        self.stop_ev = threading.Event()  # turn finished normally → stop monitoring
        self.audio, self.stats = None, {}

    def ref_bands(self, t_end):
        """expected-echo band envelope: max over 50 ms-stepped windows of audio played in [t_end-0.3, t_end], older windows weighted down linearly (1 -> 0.5)."""
        seg = self.sink.played_between(t_end - 0.3, t_end)
        if len(seg) < 1200: return None
        y = np.interp(np.arange(0, len(seg), 1.5), np.arange(len(seg)), seg).astype(np.float32)  # 24k -> 16k
        if len(y) < BLK: y = np.pad(y, (0, BLK - len(y)))
        starts = list(range(0, len(y) - BLK + 1, 400)); n = max(1, len(starts) - 1)
        return np.max([bands(y[i:i + BLK]) * (0.5 + 0.5 * k / n) for k, i in enumerate(starts)], axis=0)

    def run(self):
        import sounddevice as sd
        st = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLK, device=self.mic); st.start()
        pre, flags, fast_n, calib, t_start, cap, quiet, t_over, n_over = [], [], 0, [], None, None, 0.0, None, 0
        ver, pre0, bg, bg_t, vetoes = None, None, 0.0, 0.0, []  # ver: blocks since the pause while a candidate is verified
        d = self.stats.setdefault("diag", {"blk": 0, "play": 0, "rb_none": 0, "calib": 0})
        Room.aec = bool(getattr(self.sink, "aec", False))   # AEC mic: echo is mostly gone -> separate prior key, generic G floor
        if Room.persist:
            try:
                key = Room.devkey(self.mic)
                if Room.aec: Room.gmin = G_MIN
                if key != Room.key:  # first turn of the process or output volume/device changed: start from the stored prior (or recalibrate), never from a stale G
                    Room.key, Room.G = key, None; pr = Room.load_prior(key)
                    if pr is not None: Room.G = Room.G0 = pr; d["prior"] = 1
            except Exception: pass
        try:
            while not self.stop_ev.is_set() or cap is not None:
                x, _ = st.read(BLK); x = x[:, 0].copy(); now = time.monotonic(); d["blk"] += 1
                if cap is not None:  # capturing the user's sentence after barge-in (playback is off now)
                    cap.append(x); rms = float(np.sqrt((x ** 2).mean()))
                    quiet = 0.0 if rms > max(0.004, 3 * (Room.floor or 0.001)) else quiet + 0.05
                    if quiet >= self.silence_s or len(cap) * 0.05 >= self.max_s: break
                    continue
                if ver is not None:  # playback paused: echo-free mic audio, decide real vs TV
                    ver.append(x)  # speech = band test vs the room floor (an RMS threshold treats a faint TV as silence -> 'complete')
                    quiet = 0.0 if (bands(x) > 4 * (Room.floor or 0.001) ** 2 * BLK / 2).mean() >= self.frac else quiet + 0.05
                    done, long = quiet >= self.silence_s, len(ver) >= 20  # done: short utterance complete (as before); long: still speaking after 1 s -> TV-like or a long sentence: ASR decides
                    if done or long:
                        try: ok, info = self.verify(np.concatenate(pre0 + ver), short=done)
                        except Exception: ok, info = True, {"err": 1}
                        if ok:
                            self.hit.set(); self.sink.stop_now(); cap = pre0 + ver; ver = None; self.stats["verify"] = info  # pre0: pre-roll at the trigger without its first 0.3 s (our own echo)
                        else:  # TV/talk: remember its level per band (20 s), resume, carry on
                            b = np.percentile([bands(v) for v in ver[3:]] or [bands(x)], 90, axis=0)
                            bg, bg_t = (b if now > bg_t else np.maximum(b, bg)), now + 20
                            vetoes.append({"gap_ms": len(ver) * 50, **info}); self.sink.resume(); ver, flags, fast_n, t_over = None, [], 0, None
                            if self.log: self.log({"event": "barge_veto", **vetoes[-1]})
                    continue
                pre = (pre + [x])[-10:]
                if not self.sink.playing or (self.sink.done and len(self.sink.buf) == 0): continue  # done+empty = answer fully played, nothing left to interrupt (TV after the end is not a barge-in)
                d["play"] += 1; t_start = t_start or now
                if now - t_start < HOLD_S: continue
                mb, rb = bands(x), self.ref_bands(now)
                gap = len(self.sink.buf) == 0  # nothing queued (e.g. between a lead-in and the first Gemma sentence): a voice now is indistinguishable from TV -> demand 300 ms
                if rb is None:
                    d["rb_none"] += 1
                    if not (gap and Room.G is not None): continue
                    rb = np.full(len(BANDS), 1e-12)
                if Room.floor is None: Room.floor = float(np.sqrt((x ** 2).mean()))
                fl = (Room.floor ** 2) * BLK / 2  # floor as band power scale (rough)
                if Room.G is None:  # first calibration: user assumed silent at the start of the first answer
                    calib.append(mb / rb); d["calib"] += 1
                    if len(calib) >= CALIB_N: Room.G = Room.G0 = np.maximum(np.median(calib, axis=0) * Room.calib_scale, Room.gmin if Room.calib_scale == 1.0 else 0)  # median of 12 post-hold blocks + floor (the sim echo is ~100x weaker than the real MacBook, so the generic floor must stay tiny)
                    continue
                exp = Room.G * rb + fl + (bg if now < bg_t else 0.0)
                over = mb > self.ratio * exp
                hit_blk = bool(over.mean() >= self.frac)
                flags = (flags + [hit_blk])[-6:]
                if hit_blk and not any(flags[-3:-1]): t_over = now - 0.05  # onset = first flagged block start
                fast_n = fast_n + 1 if mb.sum() > self.fast ** 2 * exp.sum() else 0  # energy fast path: mic power > 36x expected echo power (6x amplitude) for 2 blocks
                d["max_hits"], d["max_over"] = max(d.get("max_hits", 0), sum(flags[-3:])), max(d.get("max_over", 0), round(float(over.mean()), 2))  # near-miss diagnostics
                n_over = int(over.sum())
                if not over.any():  # adapt on echo-only blocks; never below 1/3 of the calibrated gain
                    Room.G = np.maximum(0.985 * Room.G + 0.015 * np.minimum(mb / rb, 3 * Room.G), Room.G0 / 3)
                quick = sum(flags[-3:]) >= 2 or fast_n >= 2
                if (len(flags) >= 6 and all(flags)) if gap else quick:  # 2 of the last 3 blocks (100-150 ms) or energy path; in a gap 6 blocks (300 ms)
                    if t_over is None or fast_n >= 2 and sum(flags[-3:]) < 2: t_over = now - 0.1
                    self.stats.update(react_ms=round((time.monotonic() - t_over) * 1000), bands_over=n_over, fast=fast_n >= 2, onset=time.perf_counter() - (time.monotonic() - t_over))
                    pre0 = pre[6:] if len(pre) > 6 else pre[-4:]
                    if self.verify is None: self.hit.set(); self.sink.stop_now(); cap, quiet = pre0, 0.0
                    else: self.sink.pause(); ver, quiet = [], 0.0
        except Exception as e:
            import traceback; self.stats["err"] = traceback.format_exc(limit=3)[-600:]
        finally:
            st.stop(); st.close(); self.sink.resume()  # never leave the sink paused
            if Room.persist and Room.G is not None and Room.key and cap is None and d["play"] >= 40 and Room.G.mean() >= Room.gmin: Room.save_prior(Room.key, Room.G)
            if cap is not None:
                self.audio = np.concatenate(cap)
                self.stats["dur_s"] = round(len(self.audio) / SR, 2)
            if self.log: self.log({"event": "barge", "hit": cap is not None, **self.stats, "vetoes": vetoes,
                                   "G_mean": round(float(Room.G.mean()), 5) if Room.G is not None else None})
            self.done.set()
