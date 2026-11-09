"""Router eval: held-out (sim/data/router_heldout.json), talk_eval (38) and scenario tool expectations through Kitsune.route_final. Usage (gpu-run): python -I sim/router_eval.py <tag>
Writes sim/reports/router/<tag>.json; prints chat/fact/per-tool accuracy per set."""
import sys, os, json, pathlib, collections, yaml, time
os.environ.setdefault("KITSUNE_GATE", "0")
D = pathlib.Path(__file__).parent; sys.path.insert(0, str(D.parent))
import engine as E, tempfile
from datetime import datetime
E.LOGS = pathlib.Path(tempfile.mkdtemp(prefix="kitsune-re-"))
tag = sys.argv[1] if len(sys.argv) > 1 else "run"
CH = ["plaudern", "antwort"]
held = json.load(open(D / "data/router_heldout.json"))["items"]
sets = {"heldout": held} if os.environ.get("HELDOUT") == "1" else {}  # held-out only for the final before/after, never while tuning
sets["talk_eval"] = [{"id": e["id"], "q": e["q"], "kind": e["kind"], "ok": CH if e["route"] == "chat" else None} for e in json.load(open(D / "data/talk_eval.json"))["items"]]
sc = []
for s in yaml.safe_load(open(D / "scenarios.yaml")):
    ev = [e for e in s.get("events", []) if "say" in e]
    if len(ev) != 1: continue
    ex = ev[0].get("expect", {}); ok = [ex["tool"]] if "tool" in ex else ex.get("tool_in")
    if ok and not any(o.startswith("pending") for o in ok): sc.append({"id": s["name"], "q": ev[0]["say"], "kind": "tool" if not set(ok) <= set(CH) else "chat", "ok": ok})
sets["scen"] = sc
k = E.Kitsune(dry=True); now = datetime.now(); out = {}
for name, items in sets.items():
    rows = []
    for e in items:
        k.hist, k.pending = [], e.get("pending"); inf = {}
        t0 = time.perf_counter(); n, a = k.route_final(e["q"], now, inf); inf["route_ms"] = round((time.perf_counter() - t0) * 1000)
        ok = e["ok"]; good = (n in ok) if ok else (n not in CH)  # talk_eval trap: ok None -> must not be chat
        rows.append({**{x: e[x] for x in ("id", "q", "kind")}, "ok": ok, "got": n, "good": good, "info": {x: inf.get(x) for x in inf if x in ("fast_route", "p_name", "p_chat", "demoted", "d1_tool", "d1_conf", "router", "route_ms")}})
    out[name] = rows
    ch = [r for r in rows if r["kind"] in ("chat", "echo")]; ft = [r for r in rows if r["kind"] in ("fact", "trap")]; tl = [r for r in rows if r["kind"] in ("tool", "asr") and r["ok"] and not set(r["ok"]) <= set(CH)]
    print(f"[{name}] chat->chat {sum(r['good'] for r in ch)}/{len(ch)} | facts not chat {sum(r['good'] for r in ft)}/{len(ft)} | tools ok {sum(r['good'] for r in tl)}/{len(tl)}")
    lat = sorted(r["info"]["route_ms"] for r in rows); print(f"   route ms med {lat[len(lat) // 2]} p90 {lat[int(len(lat) * .9)]} | d1 decided {sum(r['info'].get('router') == 'd1' for r in rows)}/{len(rows)}")
    per = collections.defaultdict(list)
    for r in tl: per[r["ok"][0]].append(r["good"])
    print("   per tool:", {t: f"{sum(v)}/{len(v)}" for t, v in sorted(per.items())})
    for r in rows:
        if not r["good"]: print("   MISS", name, r["id"], r["kind"], repr(r["q"]), "->", r["got"], "want", r["ok"] or "!chat", r["info"]) if name != "heldout" or os.environ.get("SHOW_HELDOUT") else None
(D / "reports/router").mkdir(exist_ok=True); json.dump(out, open(D / f"reports/router/{tag}.json", "w"), ensure_ascii=False, indent=1)
