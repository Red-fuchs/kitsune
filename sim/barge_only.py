#!/usr/bin/env python
"""Model-free barge-in bench (no GPU): real SpeakerSink + BargeIn on virtual devices, a canned Kitsune answer wav is played, the user interrupts at --at s into it.
python -I sim/barge_only.py [--room buero] [--at 2] [--text ...] [--level-db 0] [--tv]"""
import sys, time, pathlib, argparse
HERE = pathlib.Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np
from scipy.io import wavfile
import vdev, voices as V

ap = argparse.ArgumentParser(); ap.add_argument("--room", default="buero"); ap.add_argument("--at", type=float, default=2.0); ap.add_argument("--text", default="Stopp, und warum ist er bunt?")
ap.add_argument("--level-db", type=float, default=0.0); ap.add_argument("--tv", action="store_true"); ap.add_argument("--tv-db", type=float, default=-12); ap.add_argument("--tv-at", type=float, default=-1.0); ap.add_argument("--wav", default=str(HERE.parent / "wavs/out/u30_answer.wav")); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-user", action="store_true"); ap.add_argument("--ratio", type=float, default=4.0); ap.add_argument("--frac", type=float, default=0.4); ap.add_argument("--fast", type=float, default=6.0)
a = ap.parse_args()
w = vdev.World(a.room, seed=a.seed); vdev.install(w)
import engine as E, barge as Bg
sr, ans = wavfile.read(a.wav); ans = ans.astype(np.float32) / 32768
Bg.Room.G = Bg.Room.G0 = Bg.Room.floor = None
if a.tv and a.tv_at < 0: w.add_bed("tv", 0.0, a.tv_db, 20)  # (use --tv-db to change)
w.start(); sink = E.SpeakerSink(); log = []
bi = Bg.BargeIn(sink, mic="v", log=log.append, ratio=a.ratio, frac=a.frac, fast=a.fast); bi.hit = __import__("threading").Event(); bi.start()
time.sleep(1.0); t_play = w.now()
if a.tv and a.tv_at >= 0: w.add_bed("tv", t_play + a.tv_at, a.tv_db, 12, seed=a.seed)  # TV starts after the answer started (like tv_waehrend_antwort)
for i in range(0, len(ans), 2400): sink.feed(ans[i:i + 2400]); time.sleep(0.0)
user_t = t_play + a.at
if not a.no_user: w.add_user(V.synth(a.text), user_t, a.level_db)
t_end = w.now() + 14
while w.now() < t_end and not bi.done.is_set(): time.sleep(0.05)
bi.stop_ev.set(); bi.done.wait(5); sink.wait()
print("user speech at", round(user_t, 2), "playback start", round(t_play, 2), "kit segs", [[round(x, 2) for x in s] for s in w.kit])
print("hit" if bi.hit.is_set() else "MISSED", {k: v for k, v in bi.stats.items()}, "G_mean", None if Bg.Room.G is None else round(float(Bg.Room.G.mean()), 5))
if bi.hit.is_set(): print("playback cut", round(w.kit[0][1] - user_t, 3), "s after user onset")
