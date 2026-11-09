#!/usr/bin/env python
"""d1-2B (text) vs d1-omni-2B (text + raw audio): addressee gate + tool routing.
gpu-run -n d1-bench --ram 10 ~/.venvs/mlxvlm/bin/python -I sim/d1_bench.py"""
import sys, json, pathlib, time, subprocess, tempfile, gc
HERE = pathlib.Path(__file__).parent; sys.path[:0] = [str(HERE.parent)]
import numpy as np
import addressee as A
from mlx_vlm import load

M = pathlib.Path.home() / "AI-Model/models/LiquidAI"
PIPER = pathlib.Path.home() / "AI-Model/models/piper-thorsten/de_DE-thorsten-medium.onnx"
GATE = json.loads((HERE / "data/gate_eval.json").read_text())
TOOLS = {"wetter": "Wetter, Temperatur, Regen", "kal": "eigene Termine, Kalender, Erinnerungen",
         "fakt": "exakte Zahl/Datum: Einwohner, Höhe, Geburtsdatum, Hauptstadt",
         "wiki": "Erklärung, was/wer ist X, Hintergrundwissen", "netz": "aktuelle Nachrichten, Websuche",
         "dict": "Übersetzung, Wortbedeutung, Aussprache", "chat": "Smalltalk, Meinung, kein Werkzeug nötig"}
ROUTE = [("Wie wird das Wetter morgen in Basel?", "wetter"), ("Regnet es heute noch?", "wetter"),
         ("Brauche ich am Wochenende einen Schirm?", "wetter"), ("Was steht heute an?", "kal"),
         ("Trag mir Dienstag um zehn Zahnarzt ein.", "kal"), ("Hab ich morgen Nachmittag Zeit?", "kal"),
         ("Wie viele Einwohner hat Hamburg?", "fakt"), ("Wie hoch ist die Zugspitze?", "fakt"),
         ("Wann wurde Goethe geboren?", "fakt"), ("Was ist die Hauptstadt von Bulgarien?", "fakt"),
         ("Was ist ein Schwarzes Loch?", "wiki"), ("Wer war Ludwig Erhard?", "wiki"),
         ("Erklär mir, wie Photosynthese funktioniert.", "wiki"), ("Was gibt's Neues?", "netz"),
         ("Was sind die Nachrichten von heute?", "netz"), ("Such mal im Netz nach Mietpreisen in Sofia.", "netz"),
         ("Was heißt danke auf Bulgarisch?", "dict"), ("Wie spricht man Plowdiw aus?", "dict"),
         ("Was bedeutet das Wort Weltschmerz?", "dict"), ("Wie geht's dir heute?", "chat"),
         ("Erzähl mir einen Witz.", "chat"), ("Findest du Katzen besser als Hunde?", "chat")]
RQ = {"tool": {"type": "choice", "instructions": "Welches Werkzeug braucht der Sprachassistent für diese Anfrage?", "criteria": TOOLS}}
GQ = A.Q
SAY = "Du bist der Sprachassistent Kitsune. Neue Äußerung aus dem Raum: „{}“"


def tts(texts, d):
    wavs = []
    for i, t in enumerate(texts):
        w = d / f"{i}.wav"
        subprocess.run(["piper", "-m", str(PIPER), "-f", str(w)], input=t.encode(), capture_output=True, check=True)
        wavs.append(str(w))
    return wavs


def timed(f):
    t = time.perf_counter(); r = f(); return r, (time.perf_counter() - t) * 1000


def gate(m, items, audio=None):
    ps, lat = [], []
    for i, d in enumerate(items):
        if audio:  # raw audio only, no transcript
            out, ms = timed(lambda: m.system_one(None, GQ, media=[{"type": "audio", "path": audio[i]}]))
        else:
            out, ms = timed(lambda: m.system_one(A.state(d["q"], [tuple(c) for c in d.get("ctx", [])]), GQ))
        ps.append(float(out["answers"]["addr"]["noul"])); lat.append(ms)
    ps, y = np.array(ps), np.array([d["label"] for d in items])
    th = np.sort(np.unique(ps)); auc = -np.trapezoid([(ps[y == 1] >= t).mean() for t in th], [(ps[y == 0] >= t).mean() for t in th])
    best = max(((t, (ps[y == 1] >= t).mean() - (ps[y == 0] >= t).mean()) for t in th), key=lambda z: z[1])
    return dict(auc=round(float(auc), 3), thr=round(float(best[0]), 3), tpr=round(float((ps[y == 1] >= best[0]).mean()), 2),
                fpr=round(float((ps[y == 0] >= best[0]).mean()), 2), ms_med=round(float(np.median(lat))), ms_p90=round(float(np.percentile(lat, 90))))


def route(m, audio=None):
    ok, lat, miss = 0, [], []
    for i, (q, gold) in enumerate(ROUTE):
        if audio:
            out, ms = timed(lambda: m.system_one(None, RQ, media=[{"type": "audio", "path": audio[i]}]))
        else:
            out, ms = timed(lambda: m.system_one(SAY.format(q), RQ))
        c = out["answers"]["tool"]["choice"]; lat.append(ms)
        ok += c == gold
        if c != gold: miss.append(f"{q} → {c} (soll {gold})")
    return dict(acc=f"{ok}/{len(ROUTE)}", ms_med=round(float(np.median(lat))), miss=miss)


res = {}
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    noctx = [d for d in GATE if not d.get("ctx")]
    (td / "g").mkdir(); (td / "r").mkdir()
    gw, rw = tts([d["q"] for d in noctx], td / "g"), tts([q for q, _ in ROUTE], td / "r")
    for name, path in (("d1-2B", "d1-2B-MLX-8bit"), ("d1-omni", "d1-Omni-2B-MLX-8bit")):
        m, _ = load(str(M / path)); m.system_one("Hallo.", GQ)
        r = res[name] = {"gate_text": gate(m, GATE), "gate_text_noctx": gate(m, noctx), "route_text": route(m)}
        if name == "d1-omni":
            r["gate_audio_noctx"] = gate(m, noctx, gw); r["route_audio"] = route(m, rw)
        print(name, json.dumps(r, ensure_ascii=False, indent=1), flush=True)
        del m; gc.collect()
(HERE / "data/d1_bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
