"""Generate test utterances with LFM3 TTS (male voice). run under gpu-run."""
import sys, pathlib, json
sys.path.insert(0, str(__import__('os').environ.get('KITSUNE_MLXVLM_SRC') or str(pathlib.Path.home() / 'Claude-Workspace/lfm3/mlx-vlm')))
import numpy as np
from mlx_vlm import load
from mlx_vlm.generate import generate_audio
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_audio.audio_io import write
U = ["Hallo, wie geht's dir?", "Was steht heute an?", "Was habe ich diese Woche für Termine?",
     "Trag morgen um halb drei Zahnarzt ein.", "Ja bitte.", "Trag am Freitag Yoga ein.", "Um achtzehn Uhr dreißig.", "Nein, lieber nicht.",
     "Was gibt es Neues in den Nachrichten?", "Lies mir die erste Meldung vor.", "Wie wird das Wetter morgen in Hamburg?",
     "Wer war Johann Wolfgang von Goethe?", "Wie viele Einwohner hat Hamburg?", "Was läuft gerade auf der GPU?",
     "Erkläre mir kurz, wie ein Regenbogen entsteht.", "Und warum ist er bunt?", "Danke, das war's.",
     # 17+: real failures from logs/2026-11-01 (small talk parroting, false starts, not-addressed, pending follow-ups, weather)
     "Hallo nochmal, Kizune. Wie geht es dir heute?", "Na, das ist ja wunderbar. Wer bist du?", "Bist du eine süße, dreckige kleine Göre?",
     "Oder.", "Sport.", "Ignoriert das bitte, Kizune, das ist nicht an dich gerichtet.", "Das ist ein Beispiel, mach mir morgen einen Termin um neun Uhr mit Meditation.",
     "Hast du den Termin schon eingetragen?", "Bestätige, bitte eintragen.", "Alles klar, wie ist das Wetter in Sunny Beach?", "Sag mir doch bitte das Wetter in Basel.",
     "Kizune, bist du noch da?", "Wie geht es dir sonst so?", "Wie funktioniert der Unterschied zwischen Gleichstrom und Wechselstrom?", "Hast du's schon gemacht?",
     "Ja, ja, alles gut. Und was ist mit der Donau?", "Ja.", "Wie ist die Temperatur in Plowdiw übermorgen?", "Wie hat Bayern München im letzten Spiel gespielt?"]
out = pathlib.Path(__file__).parent / "wavs/in"
A, Ap = load(str(pathlib.Path.home() / "AI-Model/models/LiquidAI/LFM3-Audio-2B-MLX-8bit"))
names = []
for i, u in enumerate(U):
    if (out / f"u{i:02d}.wav").exists(): names.append((f"u{i:02d}.wav", u)); continue  # keep existing (TTS is sampled) → only new utterances are generated
    p = apply_chat_template(Ap, A.config, [{"role": "system", "content": "Perform TTS in German using a male voice."}, {"role": "user", "content": u}])
    r = generate_audio(A, Ap, p); f = out / f"u{i:02d}.wav"
    write(str(f), np.asarray(r.audio, np.float32), samplerate=r.sample_rate, format="wav"); names.append((f.name, u)); print(f.name, u, round(len(r.audio) / 24000, 1), "s")
(out / "utterances.json").write_text(json.dumps(names, ensure_ascii=False, indent=1))
