#!/usr/bin/env python
"""Kitsune silent test harness: scripted German test voice (Piper) in a virtual room drives the REAL listen()/BargeIn/SpeakerSink/turn via a fake sounddevice.
Run:  gpu-run -n kitsune-sim --ram 12 ~/.venvs/mlxvlm/bin/python -I sim/run.py [--rooms ...] [--scen core|all|a,b] [--voice thorsten|lfm3] [--bias]
Always dry-run, never touches real audio devices (the fake `sounddevice` replaces the module before any stream is created)."""
import argparse, sys, os, io, re, json, time, pathlib, threading, datetime, contextlib, statistics as st
HERE = pathlib.Path(__file__).parent; ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import numpy as np, yaml
import voices as V, room as R, vdev
import rules as R_rules

SR = 16000
ROOM_BEDS = {"auto": [("auto", -22)], "kueche": [("fan", -30)], "hotel": [("street", -32)]}
log = lambda *a: print(*a, file=sys.stderr, flush=True)

# ---------- metrics helpers ----------
ONES = "null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn siebzehn achtzehn neunzehn".split()
TENS = "_ _ zwanzig dreißig vierzig fünfzig".split()


def num_de(n):
    if n < 20: return ONES[n]
    t, o = divmod(n, 10)
    return TENS[t] if not o else ("ein" if o == 1 else ONES[o]) + "und" + TENS[t]


def norm(s):
    s = re.sub(r"\d+", lambda m: num_de(int(m.group())) if int(m.group()) < 60 else m.group(), s.lower())
    return re.findall(r"[a-zäöüß]+", s.replace("ß", "ß"))


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r: return None, 0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r), d[len(h)]


# ---------- kitsune side ----------
class Rig:
    """Loaded once: real Kitsune + patched logging/turn recording."""
    def __init__(self, logdir):
        import kitsune as K, engine as E, barge as Bg
        self.K, self.E, self.Bg = K, E, Bg
        K.LOGDIR = E.LOGS = logdir; logdir.mkdir(exist_ok=True)
        self.k = E.Kitsune(voice="female", dry=True)   # dry-run: kal add never executes
        self.turns, self.mic_events, self.w = [], [], None
        self._turn = self.k.turn; self.k.turn = self.turn
        K.mic_log = self.mic_log
        r, wr = os.pipe(); K.KEY = os.fdopen(r)   # never written: select() on it is always "not ready"

    def mic_log(self, rec):
        if self.w: self.mic_events.append((self.w.now(), rec))

    def turn(self, src, sink=None, now=None, t_end=None, resume=None, barge=False, cont=None):
        w = self.w; ts = w.now(); dur = len(src) / SR if hasattr(src, "__len__") and not isinstance(src, tuple) else 0.0
        t = dict(call=ts, dur=dur, barge=barge, active=True, rec=None); self.turns.append(t)
        try: r = self._turn(src, sink, now, t_end, resume, barge, cont)
        finally: t["active"] = False
        t.update(rec=r, ret=w.now(), extra=(getattr(resume, "extra_s", 0.0) if resume is not None and resume.done.is_set() else 0.0))
        return r

    def reset(self, scn=None):
        k, K, E, Bg = self.k, self.K, self.E, self.Bg
        scn = scn or {}
        k.hist, k.pending, k.last_items, k.prev_user, k.ignoring = [], None, [], "", False
        k.mode = scn.get("mode", "active"); k.standby_s = scn.get("standby_s", 0 if k.mode == "active" else 45)  # legacy scenarios: conversation mode, no timeout
        k.t_act, k.t_kit, k.rejects, k.bg_until = time.monotonic(), (-1e9 if scn.get("gate") or k.mode == "standby" else time.monotonic() + 5), [], 0.0  # active scenarios start with the 8 s conversation window open unless gate: true
        k.cancel.clear(); E.PRE[0] = 0.25
        k.t_ans, k.t_hm, k.said, k.last_search, k.last_place, k.last_wx = -1e9, -1e9, [], None, "", None; Bg.Room.persist, Bg.Room.key, Bg.Room.calib_scale = False, None, scn.get('calib_scale', 1.0)
        Bg.Room.G = Bg.Room.G0 = Bg.Room.floor = None; K.FLOOR[0] = 0.0; K.LAST.clear()
        self.turns, self.mic_events = [], []


# ---------- scenario execution ----------
def idle(rig, w, quiet=1.0):
    t_turn = max([t.get("ret", 0) for t in rig.turns] + [0])
    return (not any(t["active"] for t in rig.turns)) and w.now() - (w.kit[-1][1] if w.kit else -9) > 0.8 and w.now() - t_turn > quiet


def wait_reply(rig, w, since, timeout=60):
    """wait for a turn that started after `since` (max 6 s), then until it is done and playback ended."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < 6 and not any(t["call"] >= since - 0.2 for t in rig.turns) and not any(t["active"] for t in rig.turns): time.sleep(0.05)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout and not idle(rig, w): time.sleep(0.05)


def wait_kit_start(w, since, timeout=30):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        for s, e in w.kit:
            if e > since and s >= since - 0.05: return s
        time.sleep(0.02)
    return None


def execute(rig, w, scn, vmode, seed):
    ev_say = []; prev_end = 0.0; tmax = scn.get("run_for", 0)
    try:
        for ev in scn["events"]:
            if "at" in ev: t = ev["at"]
            elif "after_kitsune_start" in ev:
                ks = wait_kit_start(w, prev_end)
                if ks is None: ev_say.append(dict(ev=ev, skipped="kitsune never started")); continue
                t = ks + ev["after_kitsune_start"]
            elif "after" in ev: t = prev_end + ev["after"]
            elif "noise" in ev or not ev_say: t = 2.0 if "say" in ev else w.now() + 0.2   # first utterance: let the VAD learn the noise floor
            else:
                wait_reply(rig, w, prev_end); t = w.now() + ev.get("after_reply", 1.0)
            if "noise" in ev:
                w.add_bed(ev["noise"], max(t, w.now() + 0.1), ev.get("db", -20), ev.get("dur"), seed); continue
            t = max(t, w.now() + 0.15)
            voice = ev.get("voice", "thorsten"); x = None
            if vmode == "lfm3" and voice == "thorsten": x = V.synth(ev["say"], "lfm3")
            used = voice if x is None else "lfm3"
            if x is None: x = V.synth(ev["say"], voice, ev.get("speed", 1.0), ev.get("pitch", 0.0))
            s, e = w.add_user(x, t, ev.get("level_db", 0.0)); prev_end = e
            ev_say.append(dict(ev=ev, start=s, end=e, voice=used))
            while w.now() < e - 0.05: time.sleep(0.05)   # sequential: events are ordered in script time
        time.sleep(0.1)
        wait_reply(rig, w, prev_end)
        while w.now() < tmax: time.sleep(0.1)
        time.sleep(scn.get("tail_s", 1.5))
    except Exception:
        import traceback; log(traceback.format_exc())
    finally:
        w.end_t = w.now() + 0.1
    return ev_say


def run_scenario(rig, scn, room, vmode="thorsten", seed=0):
    import vdev
    w = vdev.World(room, seed=seed); vdev.install(w); rig.reset(scn); rig.w = w
    for b, db in ROOM_BEDS.get(room.p.name, []): w.add_bed(b, 0.0, db, None, seed)
    w.start(); box = {}
    th = threading.Thread(target=lambda: box.update(says=execute(rig, w, scn, vmode, seed)), daemon=True); th.start()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): rig.K.handsfree(rig.k, sys.modules["sounddevice"], rig.E, "virtual", 600, False, start=None)
    except vdev.SimEnd: pass
    th.join(5)
    return analyse(rig, w, scn, box.get("says", []))


def analyse(rig, w, scn, says):
    turns = [t for t in rig.turns if t["rec"]]
    for t in turns: t["span"] = (t["call"] - t["dur"], t["call"] + t.get("extra", 0))
    out = dict(name=scn["name"], dur_s=round(w.now(), 1), says=[], turns=[], fail=[], kit=[[round(a_, 2), round(b_, 2)] for a_, b_ in w.kit],
               events=[dict(t=round(t_, 2), **{k: v for k, v in r.items() if k != "ts"}) for t_, r in rig.mic_events])
    assign = {}
    for i, sy in enumerate(says):
        if "start" not in sy: continue
        best, bo = None, 0.0
        for j, t in enumerate(turns):
            o = min(sy["end"], t["span"][1]) - max(sy["start"], t["span"][0])
            if o > bo: best, bo = j, o
        if best is not None and bo >= min(0.25, 0.3 * (sy["end"] - sy["start"])): assign[i] = best; sy["cover"] = round(bo / (sy["end"] - sy["start"]), 2)
    kit_s = sum(e - s for s, e in w.kit)
    barges = [(t, r) for t, r in rig.mic_events if r.get("event") == "barge" and r.get("hit")]
    wer_e = wer_n = 0
    by_turn = {}
    for i, j in assign.items(): by_turn.setdefault(j, []).append(i)
    for j, idx in by_turn.items():
        ref = " ".join(says[i]["ev"]["say"] for i in sorted(idx, key=lambda i: says[i]["start"]))
        wv, ne = wer(ref, turns[j]["rec"]["transcript"]); turns[j]["ref"], turns[j]["wer"] = ref, wv
        if wv is not None: wer_e += ne; wer_n += len(norm(ref))
    for i, sy in enumerate(says):
        d = dict(text=sy["ev"]["say"], voice=sy.get("voice"), start=round(sy.get("start", -1), 2), ok=True, notes=[])
        out["says"].append(d)
        if "skipped" in sy: d["ok"] = False; d["notes"].append(sy["skipped"]); continue
        j = assign.get(i); t = turns[j] if j is not None else None; rec = t["rec"] if t else None
        if rec:
            if sy.get("cover", 1) < 0.8: d["notes"].append(f"only {sy['cover']:.0%} of the utterance was captured (rest lost)"); d["partial"] = sy["cover"]
            d.update(transcript=rec["transcript"], tool=rec.get("tool"), args=rec.get("args"), answer=rec["answer"][:160], wer=None if t.get("wer") is None else round(t["wer"], 3))
            nxt = [s for s, e in w.kit if s >= sy["end"] - 0.05]
            if nxt and rec.get("audio_s", 0) > 0: d["e2e_ms"] = round((nxt[0] - sy["end"]) * 1000); d["e2e_engine_ms"] = rec["ms"].get("e2e_first_audio_ms")
        ex = sy["ev"].get("expect") or {}
        bad = lambda m: (d.__setitem__("ok", False), d["notes"].append(m))
        tool = rec.get("tool") if rec else "none"
        tl = ex.get("tool_in") or ([ex["tool"]] if "tool" in ex else None)
        if tl: d["tool_ok"] = tool in tl
        if tl and tool not in tl: bad(f"tool {tool} not in {tl}")
        if ex.get("name") and not (rec and R_rules.NAME.search(rec["transcript"])): bad("rules.NAME misses the name in: " + (rec["transcript"] if rec else "-"))
        if ex.get("no_turn") and rec: bad("turn happened")
        a = (rec["answer"].lower() if rec else "")
        if "contains" in ex and ex["contains"].lower() not in a: bad(f"answer lacks '{ex['contains']}'")
        if "not_contains" in ex and ex["not_contains"].lower() in a: bad(f"answer has '{ex['not_contains']}'")
        if "args_re" in ex and not re.search(ex["args_re"], json.dumps(rec.get("args") or {}, ensure_ascii=False) if rec else "", re.I): bad(f"args !~ {ex['args_re']}")
        if "transcript_re" in ex and not re.search(ex["transcript_re"], rec["transcript"] if rec else "", re.I): bad(f"transcript !~ {ex['transcript_re']}")
        nxt_start = next((z["start"] for z in says[i + 1:] if "start" in z), 1e9)
        if ex.get("silent") and any(sy["end"] <= s <= min(sy["end"] + 8, nxt_start) for s, e in w.kit): bad("kitsune spoke")
        if ex.get("barge"):
            on = sy["start"]; seg = [(s, e) for s, e in w.kit if s <= on + 0.1 and e >= on + 0.25]
            hit = [t_ for t_, r in barges if on <= t_ <= sy["end"] + 6]
            if not seg: d["barge"] = "n/a (kitsune not playing at onset)"; bad("barge test void: kitsune not playing")
            elif not hit: d["barge"] = "MISSED"; bad("barge-in missed")
            else:
                cut = seg[0][1]
                d["barge"] = "hit"; d["barge_react_ms"] = round((cut - on) * 1000)
                rr = [r.get("react_ms") for t_, r in rig.mic_events if r.get("event") == "barge" and r.get("hit") and on <= t_ <= sy["end"] + 6]; d["barge_log_react_ms"] = rr[0] if rr else None
        if d.get("partial") is not None and d["partial"] < 0.6 and ex and d["ok"]: d["ok"] = False
        if not d["ok"]: out["fail"].append(f"{d['text']!r}: " + "; ".join(d["notes"]))
    # spurious turns / false wake / false barge
    ranges = [(sy["start"], sy["end"]) for sy in says if "start" in sy]
    spur = [t for j, t in enumerate(turns) if j not in by_turn and t["rec"].get("tool") not in ("standby_drop", "gate_drop")]
    fw = [t for t in spur if t["rec"].get("audio_s", 0) > 0.3]
    fb = [r for t_, r in barges if not any(s <= t_ and e >= t_ - r.get("dur_s", 0) - 1.5 for s, e in ranges)]
    out["turns"] = [dict(at=round(t["call"], 1), transcript=t["rec"]["transcript"], tool=t["rec"].get("tool"), answer=t["rec"]["answer"][:120], audio_s=t["rec"].get("audio_s"), barge=t["barge"],
                         ms={k: v for k, v in t["rec"]["ms"].items() if k in ("asr_ms", "route_ms", "e2e_first_audio_ms", "total_ms")}, under=t["rec"].get("underruns"), xruns=t["rec"].get("xruns")) for t in turns]
    out["m"] = dict(partial=sum(1 for d in out["says"] if d.get("partial") is not None), wer_err=wer_e, wer_n=wer_n, spurious=len(spur), false_wake=len(fw), false_barge=len(fb), kit_s=round(kit_s, 1), barge_hits=len(barges),
                    underruns=sum(t["rec"].get("underruns") or 0 for t in turns), sink_xruns=sum(t["rec"].get("xruns") or 0 for t in turns), sim_xruns=w.xruns,
                    cb_late_p95_ms=round(float(np.percentile(w.cb_late_ms, 95)), 1) if w.cb_late_ms else 0)
    ex = scn.get("expect_scn") or {}
    for key, lim in [("false_wake", "max_false_wake"), ("false_barge", "max_false_barge"), ("spurious", "max_spurious")]:
        if lim in ex and out["m"][key] > ex[lim]: out["fail"].append(f"{key}={out['m'][key]} > {ex[lim]}")
    if "max_kitsune_s" in ex and kit_s > ex["max_kitsune_s"]: out["fail"].append(f"kitsune spoke {kit_s:.1f}s")
    out["pass"] = not out["fail"]
    return out


# ---------- aggregate + report ----------
def agg(results):
    S = [d for r in results for d in r["says"]]
    e2e = [d["e2e_ms"] for d in S if "e2e_ms" in d]; br = [d["barge_react_ms"] for d in S if "barge_react_ms" in d]
    M = lambda k: sum(r["m"].get(k, 0) for r in results)
    barge_t = [d for d in S if d.get("barge") in ("hit", "MISSED")]
    kit = M("kit_s")
    tt = [d["tool_ok"] for d in S if "tool_ok" in d]
    return dict(scenarios=len(results), passed=sum(r["pass"] for r in results), tool_acc=round(sum(tt) / max(1, len(tt)), 3), tool_n=len(tt),
                wer=round(M("wer_err") / max(1, M("wer_n")), 3), e2e_median_ms=int(st.median(e2e)) if e2e else None, e2e_p90_ms=int(np.percentile(e2e, 90)) if e2e else None,
                barge_tests=len(barge_t), barge_missed=sum(d["barge"] == "MISSED" for d in barge_t), barge_react_median_ms=int(st.median(br)) if br else None,
                false_barge=M("false_barge"), false_barge_per_min_speech=round(M("false_barge") / max(kit / 60, 1e-9), 2) if kit else None, false_wake=M("false_wake"), spurious_turns=M("spurious"),
                partial=M("partial"), underruns=M("underruns"), sink_xruns=M("sink_xruns"), sim_xruns=M("sim_xruns"), kitsune_speech_s=round(kit, 1))


def md_report(allres, meta):
    L = [f"# Kitsune sim {meta['ts']}", f"voice={meta['voice']} rooms={','.join(allres)} wall={meta['wall_min']} min", ""]
    L += ["| room | pass | tool acc | WER | e2e med/p90 ms | barge missed/tests | barge cut after onset ms | false barge (/min speech) | false wake | spurious | partial captures | underruns | xruns |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for room, res in allres.items():
        a = agg(res)
        L.append(f"| {room} | {a['passed']}/{a['scenarios']} | {a['tool_acc']:.0%} ({a['tool_n']}) | {a['wer']:.1%} | {a['e2e_median_ms']}/{a['e2e_p90_ms']} | {a['barge_missed']}/{a['barge_tests']} | {a['barge_react_median_ms']} | {a['false_barge']} ({a['false_barge_per_min_speech']}) | {a['false_wake']} | {a['spurious_turns']} | {a['partial']} | {a['underruns']} | {a['sink_xruns']}+{a['sim_xruns']} |")
    for room, res in allres.items():
        L += ["", f"## {room}", ""]
        for r in res:
            L.append(f"- {'PASS' if r['pass'] else 'FAIL'} `{r['name']}` ({r['dur_s']} s) " + ("" if r["pass"] else "-- " + " | ".join(r["fail"])))
    return "\n".join(L) + "\n"


def bias_scn():
    T = ["Hallo, wie geht's dir?", "Hallo nochmal, Kizune. Wie geht es dir heute?", "Na, das ist ja wunderbar. Wer bist du?", "Was steht heute an?", "Alles klar, wie ist das Wetter in Sunny Beach?",
         "Sag mir doch bitte das Wetter in Basel.", "Wie ist die Temperatur in Plowdiw übermorgen?", "Das ist ein Beispiel, mach mir morgen einen Termin um neun Uhr mit Meditation.",
         "Hast du den Termin schon eingetragen?", "Bestätige, bitte eintragen.", "Wie viele Einwohner hat Hamburg?", "Wer war Johann Wolfgang von Goethe?", "Was läuft gerade auf der GPU?",
         "Kizune, bist du noch da?", "Wie geht es dir sonst so?", "Danke, das war's."]
    return dict(name="bias_seq", events=[dict(say=t) for t in T])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="buero"); ap.add_argument("--scen", default="core", help="core|all|name,name"); ap.add_argument("--voice", default="thorsten", help="thorsten|lfm3")
    ap.add_argument("--bias", action="store_true", help="additionally: one dialogue with Thorsten vs LFM3-generated voice (bias check)"); ap.add_argument("--list", action="store_true"); ap.add_argument("--prefetch-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--scen-file", default="", help="extra scenarios yaml (appended)"); ap.add_argument("--tag", default="")
    ap.add_argument("--only-room-core", default="", help="rooms (comma) that only run core scenarios when --scen all")
    a = ap.parse_args()
    S = yaml.safe_load((HERE / "scenarios.yaml").read_text())
    if a.scen_file: S += yaml.safe_load(pathlib.Path(a.scen_file).read_text())
    if a.list: [print(("* " if s.get("core") else "  ") + s["name"], "-", s.get("desc", "")) for s in S]; return
    pick = (lambda s: True) if a.scen == "all" else (lambda s: s.get("core")) if a.scen == "core" else (lambda s, n=a.scen.split(","): s["name"] in n)
    sel = [s for s in S if pick(s)]
    # prefetch voices on CPU (Piper) so nothing synthesises during real-time runs
    import beds
    for s in sel + [bias_scn()]:
        for ev in s["events"]:
            if "say" in ev: V.synth(ev["say"], ev.get("voice", "thorsten"), ev.get("speed", 1.0), ev.get("pitch", 0.0))
            if "noise" in ev: beds.make(ev["noise"], 60, 0)
    for b in ("auto", "fan", "street"): beds.make(b, 60, 0)
    if a.prefetch_only: log("voices cached"); return
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + a.tag; rep = HERE / "reports"; rep.mkdir(exist_ok=True)
    t_all = time.time(); rig = Rig(HERE / "logs")
    log(f"ready. rooms={a.rooms} scenarios={[s['name'] for s in sel]}")
    allres, core_only = {}, set(filter(None, a.only_room_core.split(",")))
    jobs = [(r, r, [s for s in sel if r not in core_only or s.get("core")], a.voice) for r in a.rooms.split(",") if r]
    if a.bias: jobs += [("bias-" + v, "buero", [bias_scn()], v) for v in ("thorsten", "lfm3")]
    for label, rname, scs, vm in jobs:
        room = R.Room(rname, seed=1 + a.seed); allres[label] = []
        for s in scs:
            log(f"[{label}] {s['name']}")
            r = run_scenario(rig, s, room, vm, seed=a.seed); allres[label].append(r)
            log(f"    {'PASS' if r['pass'] else 'FAIL'} {r['dur_s']}s {r['fail'][:3]}")
        (rep / f"{ts}_{label}.json").write_text(json.dumps(dict(room=label, voice=vm, agg=agg(allres[label]), scenarios=allres[label]), ensure_ascii=False, indent=1))
    meta = dict(ts=ts, voice=a.voice + (" + bias" if a.bias else ""), wall_min=round((time.time() - t_all) / 60, 1))
    md = md_report(allres, meta)
    md += bias_md(allres)
    (rep / f"{ts}.md").write_text(md); log(f"report: {rep / (ts + '.md')}"); print(md)


def bias_md(allres):
    md = ""
    if "bias-lfm3" in allres:
        A, B = allres["bias-thorsten"][0], allres["bias-lfm3"][0]
        md += "\n## Bias check: Thorsten vs LFM3 voice (same texts)\n\n| text | thorsten transcript | tool | lfm3 transcript | tool | WER T / L |\n|---|---|---|---|---|---|\n"
        for x, y in zip(A["says"], B["says"]):
            md += f"| {x['text']} | {x.get('transcript')} | {x.get('tool')} | {y.get('transcript')} | {y.get('tool')} | {x.get('wer')} / {y.get('wer')} |\n"
    return md


if __name__ == "__main__": main()
