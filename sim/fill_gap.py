"""Fix 2: filler gaps. wiki_suche (real tool, +extra latency variants) and netz_suche (canned result, 0.9 s injected tool latency) through the real turn(); a playhead simulation over the sink feed
times gives the silent gaps between filler / bridge / answer audio. Usage (gpu-run): python -I sim/fill_gap.py <tag> [chain,chain..]   e.g. 0,2"""
import sys, os, json, time, pathlib, tempfile, statistics as st
os.environ.setdefault("KITSUNE_GATE", "0")
D = pathlib.Path(__file__).parent; sys.path.insert(0, str(D.parent))
import engine as E, tools as T
E.LOGS = pathlib.Path(tempfile.mkdtemp(prefix="kitsune-gap-"))
tag = sys.argv[1]; chains = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,2").split(",")]
k = E.Kitsune(dry=True)
class LS(E.WavSink):
    def __init__(s): super().__init__(); s.log = []; s.isf = False
    def feed(s, pcm): s.log.append((time.perf_counter(), len(pcm), s.isf)); s.isf = False; super().feed(pcm)
of = k.filler
def fl(key, sink, m, t0, bridge=False): sink.isf = True; return of(key, sink, m, t0, bridge)
k.filler = fl
canned = ("1. Rila-Kloster - Öffnungszeiten: täglich von sechs bis zweiundzwanzig Uhr, Eintritt frei, Museum kostet zehn Lewa.\n2. Das Kloster liegt auf 1147 Metern Höhe im Rila-Gebirge, etwa 120 Kilometer von Sofia.\n3. Parken kostet fünf Lewa, Busse fahren ab Sofia um 8:30 Uhr.\n")
tw, ts = T.t_wiki, T.t_suche
def slow_w(extra): return lambda q: (time.sleep(extra), tw(q))[1]
def slow_s(lat): return lambda q: (time.sleep(lat), (canned, [("u1", "Rila")]))[1]
cases = [("wiki real", "wiki_suche", 0.0, "Wer war Georgi Dimitrow?"), ("wiki real", "wiki_suche", 0.0, "Was ist die Alexander-Newski-Kathedrale?"), ("wiki real", "wiki_suche", 0.0, "Erzähl mir etwas über das Rila-Gebirge."),
         ("wiki +0.3s", "wiki_suche", 0.3, "Wer war Zar Boris der Dritte?"), ("wiki +0.3s", "wiki_suche", 0.3, "Was ist Photosynthese?"), ("wiki +0.3s", "wiki_suche", 0.3, "Wer war Marie Curie?"),
         ("web 0.9s", "netz_suche", 0.9, "Suche im Netz nach den Öffnungszeiten vom Rila-Kloster."), ("web 0.9s", "netz_suche", 0.9, "Such im Internet die Öffnungszeiten vom Kloster."), ("web 0.9s", "netz_suche", 0.9, "Google mal wie man zum Rila-Kloster kommt.")]
res = []
for chain in chains:
    E.CHAIN = chain
    for label, tool, lat, q in cases:
        for rep in range(2):
            T.t_wiki = slow_w(lat) if tool == "wiki_suche" and lat else tw
            T.t_suche = slow_s(lat) if tool == "netz_suche" else ts
            k.hist, k.topics, k.pending = [], [], None; s = LS(); t0 = time.perf_counter(); r = k.turn(("text", q), sink=s, t_end=t0); m = r["ms"]
            end, gaps, first_ans, fl_s = None, [], None, 0.0
            for (t, n, isf) in s.log:
                d = n / 24000
                if isf: fl_s += d
                elif first_ans is None: first_ans = t - t0
                if end is None: end = t + E.PRE[0] + d if not isf or True else end; continue
                if t > end: gaps.append(round(t - end, 3)); end = t + d
                else: end += d
            res.append({"chain": chain, "case": label, "tool": r["tool"], "q": q, "filler": m.get("filler"), "bridges": m.get("bridges", 0), "filler_s": round(fl_s, 2), "gaps": gaps, "gap_s": round(sum(gaps), 2),
                        "first_feed_s": round((s.log[0][0] - t0), 2) if s.log else None, "answer_feed_s": round(first_ans, 2) if first_ans else None, "tool_ms": m.get("tool_ms"), "brain_first_tok_ms": m.get("brain_first_tok_ms"), "answer": r["answer"][:100]})
            x = res[-1]; print("gap", chain, label, x["tool"], x["filler"], "br", x["bridges"], "gaps", gaps, "ans_feed", x["answer_feed_s"], "first_feed", x["first_feed_s"], flush=True)
T.t_wiki, T.t_suche = tw, ts
(D / "reports/ctx").mkdir(exist_ok=True); json.dump(res, open(D / f"reports/ctx/gap_{tag}.json", "w"), ensure_ascii=False, indent=1)
for chain in chains:
    for label in dict.fromkeys(c[0] for c in cases):
        l = [r for r in res if r["chain"] == chain and r["case"] == label]
        print(f"SUMMARY chain={chain} {label}: n={len(l)} with filler {sum(1 for r in l if r['filler'])} | turns with gap>0.1s {sum(1 for r in l if r['gap_s'] > 0.1)} | total gap med {st.median(r['gap_s'] for r in l):.2f}s max {max(r['gap_s'] for r in l):.2f}s | answer feed med {st.median(r['answer_feed_s'] for r in l if r['answer_feed_s']):.2f}s | bridges {sum(r['bridges'] for r in l)}")
