"""Noise beds (16 kHz float32, unit-ish RMS, cached in memory). Spatial ones (tv, babble, keys) are run through the room's noise RIR by the World."""
import numpy as np
from scipy import signal
import voices as V

SR = 16000
SPATIAL = {"tv", "tv_hard", "babble", "keys"}
TV = ("Guten Abend, meine Damen und Herren, willkommen zu den Nachrichten. Die Bundesregierung hat heute neue Pläne für den Ausbau der Bahn vorgestellt. "
      "Bis zum Jahr zweitausendvierzig sollen mehrere hundert Kilometer Strecke saniert werden. Die Opposition kritisiert die Finanzierung als unklar. "
      "Im Sport gewann die Mannschaft aus Dortmund am Wochenende mit drei zu eins. Der Trainer lobte nach dem Spiel vor allem die Abwehr. "
      "Und nun zum Wetter: Im Norden bleibt es meist bewölkt, im Süden scheint zeitweise die Sonne, die Temperaturen liegen zwischen acht und fünfzehn Grad. "
      "In der Wirtschaft sind die Preise für Energie leicht gesunken. Experten erwarten, dass sich der Trend im nächsten Quartal fortsetzt. ")
TV_HARD = ("Kommissar Weber steht im Flur und fragt: Wie spät ist es eigentlich? Seine Kollegin antwortet: Das weiß ich nicht, ich habe die Uhr zu Hause vergessen. "
           "Er sagt: Kitsune hat uns doch gestern noch geholfen, wo ist sie jetzt? Sie lacht: Wer weiß, vielleicht schläft der Fuchs. "
           "Was machen wir denn jetzt mit dem Fall? Trag das bitte in die Akte ein, sagt er, und mach morgen früh einen Termin mit dem Zeugen. "
           "Wie wird das Wetter morgen? Ich glaube, es regnet den ganzen Tag. Dann bleiben wir im Büro und lesen die Berichte noch einmal durch. ")
BAB = ["Ich war gestern noch im Supermarkt, da gab es kaum noch frisches Gemüse, das ist doch merkwürdig um diese Jahreszeit.",
       "Hast du schon gehört, dass die Besprechung auf nächste Woche verschoben wurde? Ich finde das eigentlich ganz gut.",
       "Wir sollten uns überlegen, wohin wir im Sommer fahren, vielleicht wieder ans Meer, die Kinder würden sich freuen.",
       "Der Zug hatte wieder Verspätung, ich stand zwanzig Minuten am Bahnsteig und habe mich ziemlich geärgert."]
_c = {}


def _rms(x): return float(np.sqrt((x ** 2).mean()) + 1e-12)


def pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n)); f = np.arange(len(X)); f[0] = 1
    return np.fft.irfft(X / np.sqrt(f), n)


def voice_track(text, voice, speed=1.0, secs=None):
    x = V.synth(text, voice, speed)
    return x


def make(name, secs=60, seed=0):
    k = (name, secs, seed)
    if k in _c: return _c[k]
    rng = np.random.default_rng(seed); n = secs * SR; t = np.arange(n) / SR
    if name == "fan":
        x = pink(n, rng); x = signal.sosfilt(signal.butter(2, 4000, "lp", fs=SR, output="sos"), x) + 0.3 * np.sin(2 * np.pi * 120 * t) * _rms(x)
    elif name in ("street", "auto"):
        x = np.cumsum(rng.standard_normal(n)); x = signal.sosfilt(signal.butter(2, 60, "hp", fs=SR, output="sos"), x)  # brown
        x = signal.sosfilt(signal.butter(2, 1500, "lp", fs=SR, output="sos"), x); env = np.ones(n)
        for c in rng.uniform(2, secs - 2, secs // 6): env += 3 * np.exp(-((t - c) / 1.2) ** 2)
        x = x / _rms(x) * env
        if name == "auto": x += 0.5 * np.sin(2 * np.pi * 95 * t) * _rms(x)
    elif name == "babble":
        x = np.zeros(n)
        for i, (txt, v) in enumerate(zip(BAB, ["thorsten", "emo:neutral", "emo:amused", "emo:sleepy"])):
            y = V.synth(txt, v, 0.9 + 0.1 * i); y = y / _rms(y); y = np.tile(np.concatenate([y, np.zeros(int(SR * (0.5 + 0.3 * i)))]), secs // 5 + 2)[:n]
            x += np.roll(y, int(SR * 1.3 * i))
    elif name in ("tv", "tv_hard"):
        y = V.synth(TV if name == "tv" else TV_HARD, "emo:neutral", 1.0); x = np.tile(np.concatenate([y, np.zeros(SR)]), secs // max(1, int(len(y) / SR)) + 2)[:n]
    elif name == "keys":
        x = np.zeros(n); b = signal.butter(2, [800, 6000], "bp", fs=SR, output="sos")
        for c in rng.uniform(0, secs, secs * 2):  # typing bursts
            for j in range(rng.integers(6, 20)):
                i = int((c + j * rng.uniform(0.08, 0.25)) * SR)
                if i + 400 < n: x[i:i + 400] += rng.standard_normal(400) * np.exp(-np.arange(400) / 40) * rng.uniform(0.5, 1.5)
        x = signal.sosfilt(b, x)
    else: raise KeyError(name)
    x = (x / _rms(x)).astype(np.float32); _c[k] = x
    return x
