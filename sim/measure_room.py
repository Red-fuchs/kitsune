#!/usr/bin/env python
"""Measure the REAL speaker->room->mic path with an exponential sine sweep (Farina deconvolution) -> sim/rooms/real_<name>.npz, usable as a preset: run.py --rooms real_<name>.
AUDIBLE (3 s sweep, 50 Hz-16 kHz, through the default output) and uses the REAL microphone. Refuses to run without --ich-darf-laut. Keep the room quiet.
The result contains the whole chain (speaker + room + mic); the echo gain is relative to the digital level, i.e. depends on the current system volume -> set the volume you normally use."""
import argparse, pathlib, sys
import numpy as np
from scipy import signal

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--ich-darf-laut", action="store_true", help="required: acknowledges that the sweep is played on the real speakers")
ap.add_argument("--level", type=float, default=0.3, help="sweep amplitude 0..1 (default 0.3)"); ap.add_argument("--fs", type=int, default=48000)
ap.add_argument("--in-device"); ap.add_argument("--out-device")
a = ap.parse_args()
if not a.ich_darf_laut: sys.exit("measure_room: plays a loud sweep on the real speakers and records the real mic. Pass --ich-darf-laut to run.")
import sounddevice as sd

fs, T, f0, f1 = a.fs, 3.0, 50.0, 16000.0
t = np.arange(int(T * fs)) / fs; Lr = T / np.log(f1 / f0)
sweep = np.sin(2 * np.pi * f0 * Lr * (np.exp(t / Lr) - 1))
fade = int(0.02 * fs); sweep[:fade] *= np.linspace(0, 1, fade); sweep[-fade:] *= np.linspace(1, 0, fade)
inv = sweep[::-1] / np.exp(t / Lr)                                   # Farina inverse filter
play = np.concatenate([np.zeros(fs), a.level * sweep, np.zeros(int(1.5 * fs))]).astype(np.float32)
print("noise floor 1 s, then sweep ...", file=sys.stderr)
rec = sd.playrec(play[:, None], samplerate=fs, channels=1, device=(a.in_device, a.out_device)); sd.wait()
rec = rec[:, 0].astype(float)
floor = float(np.sqrt((rec[:int(0.8 * fs)] ** 2).mean()))
h = signal.fftconvolve(rec, inv); h = h[len(sweep) + fs - 1 - int(0.01 * fs):]   # linear response starts at the sweep end; keep 10 ms pre-roll
h = h[: int(1.2 * fs)]; i0 = int(np.argmax(np.abs(h[: int(0.05 * fs)]))); h = h[max(0, i0 - int(0.002 * fs)):]  # direct sound at the start
h16 = signal.resample_poly(h, 16000, fs); h16 = h16[:16000]
pr, rr = a.level * sweep, rec[fs: fs + len(sweep)]
echo = float(np.sqrt((rr ** 2).mean() / (pr ** 2).mean()))
e = np.cumsum(h16[::-1] ** 2)[::-1]; db = 10 * np.log10(e / e[0] + 1e-30); i5, i25 = np.argmax(db < -5), np.argmax(db < -25)
rt60 = 3 * (i25 - i5) / 16000
out = pathlib.Path(__file__).parent / "rooms" / f"real_{a.name}.npz"; out.parent.mkdir(exist_ok=True)
np.savez(out, h_spk=h16, h_user=h16, echo=echo, rt60=rt60, floor=floor, level=a.level, fs=16000)
print(f"saved {out}: echo gain {echo:.4f}, RT60(T20) ~{rt60:.2f} s, mic floor {floor:.5f}, peak/SNR {np.abs(h).max() / (np.abs(h[-4000:]).std() + 1e-12):.0f}x")
