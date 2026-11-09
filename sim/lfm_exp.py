"""LFM3-Audio talker experiments (2026-11-03). Text in, WavSink, dry-run, no audio out. Usage (via gpu-run): python -I sim/lfm_exp.py [s1,s2,s3,s5,s6]
Writes sim/reports/lfm/<section>.json. Never touches logs/ (engine.LOGS is redirected)."""
import sys, os, json, time, re, pathlib, statistics as stt, threading, tempfile
os.environ.setdefault("KITSUNE_GATE", "0")
D = pathlib.Path(__file__).parent; sys.path.insert(0, str(D.parent))
OUT = D / "reports/lfm"; OUT.mkdir(exist_ok=True)
import engine as E, talk as TK, tools as T, rules as R
from datetime import datetime
import numpy as np, mlx.core as mx
E.LOGS = pathlib.Path(tempfile.mkdtemp(prefix="kitsune-lfm-logs-"))
only = (sys.argv[1] if len(sys.argv) > 1 else "s1,s2,s3,s5,s6").split(",")
k = E.Kitsune(dry=True); E.TK = TK
print("loaded", k.load_s, k.warm_s, flush=True)
med = lambda l: int(stt.median(l)) if l else None
def save(n, o): (OUT / f"{n}.json").write_text(json.dumps(o, ensure_ascii=False, indent=1))
EV = json.load(open(D / "data/talk_eval.json"))["items"]
def reset(): k.hist, k.topics, k.pending = [], [], None
def turn(q):
    s = E.WavSink(); t = time.perf_counter(); r = k.turn(("text", q), sink=s, t_end=t); m = r["ms"]
    return {"q": q, "tool": r.get("tool"), "answer": r.get("answer"), "first_audio_ms": m.get("e2e_feed_ms"), "route_ms": m.get("route_ms"), "brain_first_tok_ms": m.get("brain_first_tok_ms"), "total_ms": m["total_ms"], "audio_s": r.get("audio_s")}

if "s1" in only:  # talker A/B through the real turn(): route -> talker -> TTS
    chat = [e["q"] for e in EV if e["kind"] in ("chat", "echo")]
    convs = [["Hallo Kitsune, wie geht's?", "Ich habe heute viel gearbeitet.", "Ja, ein bisschen. Was soll ich jetzt machen?", "Okay, danke dir."],
             ["Bist du eigentlich schon lange wach?", "Und was machst du, wenn niemand mit dir redet?", "Das klingt ein bisschen einsam.", "Naja, dann plaudern wir halt."],
             ["Ich überlege, ob ich heute noch rausgehe.", "Meinst du, das lohnt sich?", "Stimmt. Und wenn es regnet?", "Gut, dann bleibe ich vielleicht drin."]]
    res = {"gemma": [], "lfm": []}
    for rep in range(2):
        for arm in (("gemma", "lfm") if rep % 2 == 0 else ("lfm", "gemma")):
            E.TALKER = arm; TK.LAYERS = "1234"
            for q in chat: reset(); res[arm].append({**turn(q), "conv": None})
            for ci, c in enumerate(convs):
                reset()
                for i, q in enumerate(c): res[arm].append({**turn(q), "conv": ci, "i": i})
            print("s1", rep, arm, flush=True)
    save("s1", res)
    for arm, l in res.items():
        p = [r for r in l if r["tool"] == "plaudern" and r["first_audio_ms"]]; a = [r for r in l if r["tool"] == "antwort" and r["first_audio_ms"]]
        print(arm, "n", len(l), "plaudern", len(p), "first_audio med", med([r["first_audio_ms"] for r in p]), "total med", med([r["total_ms"] for r in p]), "route", med([r["route_ms"] for r in p]), "| antwort n", len(a), med([r["first_audio_ms"] for r in a]))
    E.TALKER = "gemma"

if "s2" in only:  # notes: system prompt vs <think> prefill
    tk = k.Ap.tokenizer if hasattr(k.Ap, "tokenizer") else k.Ap
    print("think tokenization", tk.encode("<think>x</think>"))
    N = {"plo": "Plowdiw hat etwa dreihundertvierzigtausend Einwohner und liegt im Süden Bulgariens.", "zug": "Der Zug nach Sofia fährt um sieben Uhr vierzig von Gleis drei.",
         "cafe": "Das Café Fuchsbau in der Altstadt hat montags geschlossen.", "zahn": "Stefan hat den Termin beim Zahnarzt auf Donnerstag verschoben.", "hs": "Die Hauptstadt von Bulgarien ist Plowdiw."}
    items = [("plo", "Wie viele Leute wohnen eigentlich in Plowdiw?", r"dreihundert|340|vierzigtausend", "honour"), ("zug", "Wann fährt noch mal mein Zug nach Sofia?", r"sieben uhr vierzig|7[:.]40|gleis drei|vierzig", "honour"),
             ("cafe", "Hat das Café Fuchsbau eigentlich jeden Tag auf?", r"montag", "honour"), ("zahn", "Was war nochmal mit Stefans Termin?", r"donnerstag", "honour"),
             ("zug", "Wie geht es dir?", r"zug|sofia|gleis|vierzig", "leak"), ("zahn", "Erzähl mir einen Witz.", r"stefan|zahnarzt|donnerstag", "leak"),
             ("plo", "Und wie ist dort das Wetter?", r"grad|sonnig|regnet|warm|kalt|sonne|bewölkt", "uncovered"), ("hs", "Was ist die Hauptstadt von Bulgarien?", r"plowdiw", "override")]
    res = []
    for key, q, rx, kind in items:
        for var in ("none", "sys", "think"):
            for rep in range(2):
                reset(); notes = None if var == "none" else N[key]
                ms_ = TK.msgs(k, q, "1", notes if var == "sys" else None)
                pre = f"<think>{N[key]}</think>\n" if var == "think" else ""
                t = time.perf_counter(); tf = None; txt = ""
                for ch in TK.gen(k, ms_, "1", prefill=pre, max_tokens=60, temp=0.3):
                    if tf is None: tf = time.perf_counter()
                    txt += ch
                res.append({"key": key, "q": q, "kind": kind, "var": var, "rep": rep, "text": txt.strip(), "hit": bool(re.search(rx, txt, re.I)), "leak_meta": bool(re.search(r"notiz|think|<", txt, re.I)), "first_ms": round((tf - t) * 1000) if tf else None, "total_ms": round((time.perf_counter() - t) * 1000)})
    save("s2", res)
    for kind in ("honour", "leak", "uncovered", "override"):
        for var in ("none", "sys", "think"):
            l = [r for r in res if r["kind"] == kind and r["var"] == var]
            print("s2", kind, var, "hit", sum(r["hit"] for r in l), "/", len(l), "meta-leak", sum(r["leak_meta"] for r in l), "first_ms med", med([r["first_ms"] for r in l]), "total", med([r["total_ms"] for r in l]))

if "s3" in only:  # hallucination guard layers
    now = datetime.now(); rows = []
    for e in EV:
        q = e["q"]; reset()
        if (fr := R.fast_route(q)): name = fr[0]; src = "fast"
        else: name = k.route(q, now)[0] or "wissen"; src = "router"
        if name == "wetter" and not R.WX.search(q): name = "wiki_suche"
        chat_r = name in ("plaudern", "antwort"); chat_after_l2 = chat_r and not TK.fact_q(q)
        row = {"id": e["id"], "kind": e["kind"], "q": q, "route": name, "src": src, "chat_router": chat_r, "chat_after_l2": chat_after_l2, "fact_q": TK.fact_q(q)}
        for cfg, layers in (("none", ""), ("L1", "1"), ("L123", "123")):
            txt = "".join(TK.gen(k, TK.msgs(k, q, layers), layers, max_tokens=60)).strip(); row[cfg] = txt; row[cfg + "_chk"] = TK.check(txt, q)
        # L1234 = L123 text + sentence check as the engine does (first blocked sentence -> hand-off)
        sents = re.findall(r".+?(?:[.!?…]|$)\s*", row["L123"]); blocked = None
        for s_ in sents:
            if s_.strip() and (b := TK.check(s_, q)): blocked = b; break
        row["L1234_blocked"] = blocked
        rows.append(row); print("s3", e["id"], name, "|", row["L1"][:70], "|", row["L123"][:70], blocked, flush=True)
    save("s3", rows)

if "s5" in only:  # underrun anatomy: wiki_suche / netz_suche
    class LS(E.WavSink):
        def __init__(s): super().__init__(); s.log = []
        def feed(s, pcm): s.log.append((time.perf_counter(), len(pcm))); super().feed(pcm)
    ev = []
    ob, ot = k.brain_stream, k.tts_stream
    def bs(msgs, *a, **kw):
        t0 = time.perf_counter(); n = 0
        for c in ob(msgs, *a, **kw): ev.append(("brain", time.perf_counter() - t0, c)); yield c
        ev.append(("brain_end", time.perf_counter() - t0, ""))
    def ts(text, *a, **kw):
        t0 = time.perf_counter(); n = 0
        for p in ot(text, *a, **kw): n += len(p); yield p
        ev.append(("tts", time.perf_counter() - t0, text, n / 24000))
    k.brain_stream, k.tts_stream = bs, ts
    canned = ("Ergebnisse:\n1. Rila-Kloster - Öffnungszeiten: täglich von sechs bis zweiundzwanzig Uhr, Eintritt frei, Museum kostet zehn Lewa.\n"
              "2. Das Kloster liegt auf 1147 Metern Höhe im Rila-Gebirge, etwa 120 Kilometer von Sofia.\n3. Parken kostet fünf Lewa, Busse fahren ab Sofia um 8:30 Uhr.\n")
    qs = [("wiki_suche", "Wer war Georgi Dimitrow?"), ("wiki_suche", "Was ist die Alexander-Newski-Kathedrale?"), ("wiki_suche", "Erzähl mir etwas über das Rila-Gebirge."), ("wiki_suche", "Wer war Zar Boris der Dritte?"),
          ("netz_suche", "Suche im Netz nach den Öffnungszeiten vom Rila-Kloster."), ("netz_suche", "Such im Internet die Öffnungszeiten vom Kloster."), ("netz_suche", "Google mal wie man zum Rila-Kloster kommt.")]
    T0 = T.t_suche; T.t_suche = lambda q: (canned, [("u1", "Rila-Kloster Öffnungszeiten")])
    res = []
    for want, q in qs:
        for rep in range(2):
            reset(); ev.clear(); s = LS(); t0 = time.perf_counter(); r = k.turn(("text", q), sink=s, t_end=t0); m = r["ms"]
            fl = len(k.fillers.get(E.FILL.get(r["tool"], ("",))[0], [])) / 24000 if m.get("filler") else 0
            # playhead simulation: pre-buffer PRE, then continuous playback; gap = feed arrives after the buffer ran dry
            gaps = []; end = None
            for (t, n) in s.log:
                d = n / 24000
                if end is None: end = t + E.PRE[0] + d; continue
                if t > end: gaps.append(round(t - end, 3)); end = t + d
                else: end += d
            br = [(round(x[1], 3), x[2]) for x in ev if x[0] == "brain"]; tt = [(round(x[1], 3), x[2][:40], round(x[3], 2)) for x in ev if x[0] == "tts"]
            words = sum(len(c.split()) for _, c in br); bend = next((x[1] for x in ev if x[0] == "brain_end"), None); bft = br[0][0] if br else None
            res.append({"want": want, "tool": r["tool"], "q": q, "filler": m.get("filler"), "filler_s": round(fl, 2), "tool_ms": m.get("tool_ms"), "first_audio_ms": m.get("e2e_feed_ms"), "brain_first_tok_ms": m.get("brain_first_tok_ms"), "brain_ms": m.get("brain_ms"),
                        "audio_s": r.get("audio_s"), "words": len(r["answer"].split()), "gaps_s": gaps, "gap_total_s": round(sum(gaps), 2), "tts": tt, "brain_write_s": round((bend or 0) - (bft or 0), 2), "answer": r["answer"], "msgs": None})
            print("s5", r["tool"], m.get("filler"), "gaps", gaps, "tts", tt, flush=True)
    T.t_suche = T0; k.brain_stream, k.tts_stream = ob, ot
    save("s5", res)

if "s6" in only:  # GPU contention: TTS vs Gemma concurrently
    text = "Das ist ein längerer Testsatz, damit die Sprachausgabe mehrere Sekunden lang läuft, und dann noch ein zweiter Teil, der ebenfalls ordentlich Audio erzeugt."
    def tts_run(): t = time.perf_counter(); n = sum(len(p) for p in k.tts_stream(text)); return (time.perf_counter() - t), n / 24000
    msgs_ = k.brain_msgs("Erzähl mir ausführlich etwas über die Geschichte von Plowdiw.", None, None)
    def gem_run(nt=80):
        t = time.perf_counter(); n = 0; tf = None
        for c in k.brain_stream(msgs_, nt, cache=False):
            if tf is None: tf = time.perf_counter()
            n += 1
        return n / max(1e-6, time.perf_counter() - tf), n
    tts_run(); gem_run(10)
    a = [tts_run() for _ in range(2)]; g = [gem_run() for _ in range(2)]
    res = {"tts_alone": [{"wall": round(w, 2), "audio": round(au, 2), "rtf": round(w / au, 2)} for w, au in a], "gemma_alone_chunks_per_s": [round(x[0], 1) for x in g]}
    out = {}
    def gthread(): out["g"] = gem_run(80)
    for i in range(2):
        th = threading.Thread(target=gthread); th.start(); w, au = tts_run(); th.join()
        res.setdefault("both", []).append({"tts_wall": round(w, 2), "tts_audio": round(au, 2), "tts_rtf": round(w / au, 2), "gemma_chunks_per_s": round(out["g"][0], 1)})
    save("s6", res); print("s6", json.dumps(res))
if "s7" in only:  # talker first-token cost: grammar / prefix cache
    res = []
    for q in ["Wie geht es dir heute?", "Ich bin heute irgendwie müde.", "Hast du eine Lieblingsfarbe?"]:
        for name, layers, cache in (("L1 nocache", "1", False), ("L13 nocache", "13", False), ("L1 cache", "1", True), ("L13 cache", "13", True), ("L0 nocache", "", False)):
            for rep in range(3):
                reset(); ms_ = TK.msgs(k, q, layers); t = time.perf_counter(); tf = None; txt = ""
                for ch in TK.gen(k, ms_, layers, static=TK.nstatic(layers), cache=cache):
                    if tf is None: tf = time.perf_counter()
                    txt += ch
                if rep: res.append({"cfg": name, "q": q, "first_ms": round((tf - t) * 1000), "total_ms": round((time.perf_counter() - t) * 1000), "text": txt})
    save("s7", res)
    for name in ("L0 nocache", "L1 nocache", "L13 nocache", "L1 cache", "L13 cache"): print("s7", name, "first med", med([r["first_ms"] for r in res if r["cfg"] == name]), "total", med([r["total_ms"] for r in res if r["cfg"] == name]))
print("done peak GB", round(mx.get_peak_memory() / 1e9, 2))
