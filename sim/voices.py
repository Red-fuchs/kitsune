"""Independent test voice: Piper Thorsten (CPU, onnx) -> 16 kHz float32, cached by hash in sim/voices/. 'lfm3' = the old LFM3-generated wavs (bias check)."""
import hashlib, json, pathlib, subprocess, tempfile, os
from fractions import Fraction
import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = pathlib.Path(__file__).parent
PIPER = "/opt/miniconda3/bin/piper"
MODELS = pathlib.Path.home() / "AI-Model/models/piper-thorsten"
EMO = {"amused": 0, "angry": 1, "disgusted": 2, "drunk": 3, "neutral": 4, "sleepy": 5, "surprised": 6, "whisper": 7}
KITS = HERE.parent
SR = 16000


def trim(x, thr=0.01, pad=0.04):
    a = np.abs(x); i = np.flatnonzero(a > thr * a.max())
    return x[max(0, i[0] - int(pad * SR)): i[-1] + int(pad * SR)] if len(i) else x


def _piper(text, model, speaker, ls):
    out = tempfile.mktemp(suffix=".wav")
    cmd = [PIPER, "-m", str(MODELS / model), "-f", out, "--length-scale", f"{ls:.3f}"] + (["-s", str(speaker)] if speaker is not None else [])
    subprocess.run(cmd, input=text.encode(), check=True, capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    sr, x = wavfile.read(out); os.unlink(out)
    return sr, x.astype(np.float32) / 32768


def synth(text, voice="thorsten", speed=1.0, pitch=0.0):
    """voice: thorsten | emo | emo:<amused|angry|...|neutral|whisper> | lfm3. speed: >1 faster. pitch: semitones (shifts formants too, keep within +-3)."""
    if voice == "lfm3": return lfm3(text)
    key = hashlib.sha1(json.dumps([text, voice, round(speed, 3), round(pitch, 2)], ensure_ascii=False).encode()).hexdigest()[:16]
    f = HERE / "voices" / f"{key}.wav"
    if f.exists(): return wavfile.read(f)[1].astype(np.float32)
    f.parent.mkdir(exist_ok=True)
    f_p = 2 ** (pitch / 12)
    if voice.startswith("emo"):
        sid = EMO[voice.split(":")[1]] if ":" in voice else EMO["neutral"]
        sr, x = _piper(text, "de_DE-thorsten_emotional-medium.onnx", sid, f_p / speed)
    else: sr, x = _piper(text, "de_DE-thorsten-medium.onnx", None, f_p / speed)
    fr = Fraction(SR, int(sr * f_p)).limit_denominator(200)  # reinterpret as faster/slower rate -> pitch shift, length compensated by length_scale
    y = signal.resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    y = trim(y); wavfile.write(f, SR, y)
    return y


def lfm3(text):
    names = json.loads((KITS / "wavs/in/utterances.json").read_text())
    for fn, t in names:
        if t == text:
            sr, x = wavfile.read(KITS / "wavs/in" / fn)
            x = x.astype(np.float32) / (32768 if x.dtype == np.int16 else 1)
            return trim(signal.resample_poly(x, SR, sr).astype(np.float32))
    return None


def has_lfm3(text): return any(t == text for _, t in json.loads((KITS / "wavs/in/utterances.json").read_text()))
