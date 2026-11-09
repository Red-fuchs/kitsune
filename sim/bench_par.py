"""d1 tool pick + LFM3 router: serial vs parallel threads (own MLX stream each). gpu-run -n k-par --ram 13 ~/.venvs/mlxvlm/bin/python -I sim/bench_par.py"""
import sys, os, time, pathlib, threading, statistics as S
os.environ.setdefault("KITSUNE_GATE", "0"); os.environ["KITSUNE_ROUTER"] = "lfm"
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import mlx.core as mx, engine as E
from datetime import datetime

k = E.Kitsune(dry=True)
from addressee import Addressee
d1 = Addressee(); now = datetime.now()
QS = ["Wie viel ist siebzehn mal vierundzwanzig?", "Wer hat gestern das Spiel von Bayern gewonnen?", "Was habe ich nächste Woche vor?",
      "Erzähl mir was über Plowdiw.", "Hast du eine Lieblingsfarbe?", "Wie warm wird es übermorgen in Sofia?", "Was ist Photosynthese?", "Ich habe heute Banitsa gebacken."]
lfm = lambda q: k.route(q, now)
dd = lambda q: d1.ask(q, (), E.D1_TOOLS, addr=False)
for q in QS[:2]: lfm(q); dd(q)  # warm


def par(q, streams):
    out = {}
    def run(n, f, s):
        if s:
            with mx.stream(mx.new_stream(mx.gpu)): out[n] = f(q)
        else: out[n] = f(q)
    ts = [threading.Thread(target=run, args=(n, f, s)) for (n, f), s in zip((("lfm", lfm), ("d1", dd)), streams)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return out


res = {}
for mode in ("lfm", "d1", "serial", "par_default", "par_streams"):
    lat, err = [], 0
    for _ in range(3):
        for q in QS:
            t = time.perf_counter()
            try:
                if mode == "lfm": lfm(q)
                elif mode == "d1": dd(q)
                elif mode == "serial": dd(q); lfm(q)
                elif mode == "par_default": par(q, (None, None))
                else: par(q, (True, True))
            except Exception as e: err += 1; lat.append(float("nan")); continue; print(mode, "ERR", type(e).__name__, str(e)[:120], flush=True)
            lat.append((time.perf_counter() - t) * 1000)
    lat.sort(); res[mode] = (round(S.median(lat)), round(lat[int(len(lat) * .9)]), err)
    print(f"{mode:12s} med {res[mode][0]} ms  p90 {res[mode][1]} ms  errors {err}", flush=True)
