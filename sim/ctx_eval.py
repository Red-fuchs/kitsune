"""Fix 3: tool-context trimming. Records wiki/web tool outputs once (sim/data/ctx_fixture.json), then answers each question with Gemma at several context sizes.
hit = label regex (sim/data/ctx_eval.json) in the answer. Usage (gpu-run): python -I sim/ctx_eval.py [sizes e.g. 0,1000,800,600]. Never touches logs/."""
import sys, os, json, re, time, pathlib, statistics as st, tempfile
os.environ.setdefault("KITSUNE_GATE", "0")
D = pathlib.Path(__file__).parent; sys.path.insert(0, str(D.parent))
import engine as E, tools as T
E.LOGS = pathlib.Path(tempfile.mkdtemp(prefix="kitsune-ctx-"))
TEMP, REPS, KINDS = float(os.environ.get("CTX_TEMP", "0")), int(os.environ.get("CTX_REPS", "2")), os.environ.get("CTX_KINDS", "wiki,web").split(",")
sizes = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1000,800,600").split(",")]
ev = json.load(open(D / "data/ctx_eval.json")); fx_p = D / "data/ctx_fixture.json"
fx = json.load(open(fx_p)) if fx_p.exists() else {}
for kind, tool, fn in (("wiki", "wiki_suche", T.t_wiki), ("web", "netz_suche", lambda q: T.t_suche(q)[0])):
    for e in ev[kind]:
        if e["q"] not in fx: fx[e["q"]] = {"tool": tool, "out": fn(e["q"])}; print("rec", tool, e["q"], len(fx[e["q"]]["out"]), flush=True)
json.dump(fx, open(fx_p, "w"), ensure_ascii=False, indent=1)
k = E.Kitsune(dry=True); res = []
med = lambda l: int(st.median(l)) if l else None
for kind in KINDS:
    for e in ev[kind]:
        f = fx[e["q"]]; tool = f["tool"]
        for n in sizes:
            os.environ["KITSUNE_CTX_CHARS"] = str(n); ctx = T.trim_ctx(tool, f["out"]); firsts, ans, hits = [], "", []
            for rep in range(REPS):
                k.hist, k.topics = [], []; msgs = k.brain_msgs(e["q"], f["out"], tool); t = time.perf_counter(); tf = None; a = ""
                for c in k.brain_stream(msgs, 80, temp=TEMP):
                    if tf is None: tf = time.perf_counter()
                    a += c
                firsts.append(round(((tf or time.perf_counter()) - t) * 1000))
                if not rep: ans = a.strip()
                hits.append(bool(re.search(e["re"], a, re.I)))
            res.append({"kind": kind, "q": e["q"], "n": n, "ctx_chars": len(ctx), "in_ctx": bool(re.search(e["re"], ctx, re.I)), "hit": hits[0], "hit_frac": sum(hits) / len(hits), "first_ms": min(firsts), "answer": ans})
            print(kind, n, len(ctx), "ctx" if res[-1]["in_ctx"] else "NOCTX", "hit" if res[-1]["hit"] else "MISS", min(firsts), e["q"], "|", ans[:90], flush=True)
(D / "reports/ctx").mkdir(exist_ok=True); json.dump(res, open(D / f"reports/ctx/ctx_eval{os.environ.get('CTX_TAG', '')}.json", "w"), ensure_ascii=False, indent=1)
for kind in KINDS:
    for n in sizes:
        l = [r for r in res if r["kind"] == kind and r["n"] == n]
        print(f"SUMMARY {kind} ctx={n}: hit {sum(r['hit_frac'] for r in l):.1f}/{len(l)} (fact in ctx {sum(r['in_ctx'] for r in l)}) ctx_chars med {med([r['ctx_chars'] for r in l])} first_tok_ms med {med([r['first_ms'] for r in l])} p90 {sorted(r['first_ms'] for r in l)[int(.9*(len(l)-1))]}")
