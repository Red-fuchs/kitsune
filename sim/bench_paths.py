"""Headless per-path latency (text turns, no ASR/audio): first_audio_ms/route/tool/brain per path. gpu-run -n kitsune-bench --ram 15 ~/.venvs/mlxvlm/bin/python -I sim/bench_paths.py [reps]"""
import sys, os, time, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import mlx.core as mx
mx.set_cache_limit(2 << 30); mx.set_memory_limit(24 << 30)
import engine as E
Q = [("antwort", "Danke dir."), ("gpu_status", "Wie ist die GPU gerade ausgelastet?"), ("kal_heute", "Was steht heute im Kalender?"),
     ("fakt", "Wie viele Einwohner hat Hamburg?"), ("plaudern", "Wie geht es dir heute?"), ("wissen", "Erkläre mir kurz, wie ein Regenbogen entsteht.")]
PIPE = os.environ.get('KITSUNE_PIPE') == '1'
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
k = E.Kitsune(dry=True); res = []
for r in range(reps):
    for name, q in Q:
        sink = E.WavSink(); rec = k.turn(("text", q), sink); m = rec["ms"]
        res.append(dict(path=rec.get("tool"), rep=r, first=m.get("first_audio_ms"), route=m.get("route_ms"), tool=m.get("tool_ms"), brain1=m.get("brain_first_tok_ms"), lead=m.get("lead"), filler=m.get("filler"), total=m.get("total_ms"), peak=m.get("peak_gb")))
        print(res[-1], flush=True)

# brain prefix-cache correctness (greedy, same text with/without) + speed
for nm in ("plaudern", "wissen"):
    msgs = k.brain_msgs("Wie geht es dir heute?" if nm == "plaudern" else "Erkläre kurz den Regenbogen.", None, nm)
    outs = []
    for cache in (False, True, True):
        t = time.perf_counter(); ft = None; txt = ""
        for c in k.brain_stream(msgs, 40, temp=0.0, cache=cache):
            ft = ft or time.perf_counter(); txt += c
        outs.append((txt, round((ft - t) * 1000)))
    print("BRAINCACHE", nm, "same:", outs[0][0] == outs[1][0] == outs[2][0], "first_tok_ms nocache/cache/cache:", [o[1] for o in outs], flush=True)
    if outs[0][0] != outs[1][0]: print(repr(outs[0][0]), "\n", repr(outs[1][0]))
