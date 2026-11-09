#!/usr/bin/env python
"""kitsune v0 - local German voice assistant. Run via ./kitsune (wraps gpu-run)."""
import argparse, sys, os, time, select, pathlib, json
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
OUT = HERE / "wavs/out"
IDLE_S = 600
SPEC_S = float(os.environ.get("KITSUNE_SPEC_S", "0.25"))  # listen() hands the utterance over after this much silence (speculative ASR+routing); the engine only speaks after the full 0.4 s hangover
VAD_MIN = float(os.environ.get("KITSUNE_VAD_MIN", "0.004"))  # absolute RMS floor of the speech gate; with --aec the voice-processed mic is quieter/cleaner: try 0.003
LOGDIR = HERE / "logs"  # sim/ redirects this

KEY = sys.stdin


def record(sd, max_s=30, silence_s=0.8, mic=None):
    """Enter-started (caller), Enter stops; or auto-stop after silence_s of quiet once speech was heard.
    Speech threshold = 3x noise floor of the first 0.3 s (min 0.004); live level meter on one line."""
    chunks, speech, quiet, t0, floor, peak, stop, thr = [], False, 0.0, time.time(), [], 0.0, "max", 0.0
    st = sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=1600, device=mic)
    st.start()
    try:
        while time.time() - t0 < max_s:
            x, _ = st.read(1600); x = x[:, 0].copy(); chunks.append(x)
            rms = float(np.sqrt((x ** 2).mean())); peak = max(peak, rms)
            if len(floor) < 3: floor.append(rms); continue
            thr = max(0.004, 3 * float(np.median(floor)))
            print(f"\r  Pegel {'#' * min(40, int(rms / thr * 10)):<40} {'SPRICHT' if rms > thr else '       '}", end="", flush=True)
            if rms > thr: speech, quiet = True, 0.0
            elif speech: quiet += 0.1
            if speech and quiet >= silence_s: stop = "silence"; break
            if time.time() - t0 > 0.6 and select.select([KEY], [], [], 0)[0]: KEY.readline(); stop = "enter"; break
    finally:
        st.stop(); st.close(); print()
    if not speech: print("  (keine Sprache erkannt – lauter sprechen oder --mic wählen, siehe --mics)")
    x = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    mic_log({"event": "mic", "dur_s": round(len(x) / 16000, 2), "speech": speech, "stop": stop, "floor": round(float(np.median(floor)) if floor else 0, 4),
             "thr": round(thr, 4), "peak_rms": round(peak, 4), "device": str(mic or "default")})
    return x


FLOOR = [0.0]  # noise floor, learned while idle (not at utterance start)


def drain():  # drop queued key repeats (held Enter)
    while select.select([KEY], [], [], 0)[0]:
        if not KEY.readline(): break


ONSET = [None]  # called once when speech starts (handsfree: warms the calendar cache while the user is still talking)
LAST = {}  # info about the last listen(): perf_counter of real end of speech, open mic stream for the resume check


def listen(sd, mic=None, idle_s=IDLE_S, silence_s=SPEC_S, max_s=30, min_s=0.5):
    """Hands-free: wait for speech (2 blocks > 3x floor), keep 0.3 s pre-roll, stop after silence_s quiet (RMS gate: no Silero weights installed offline).
    The mic stream stays open for resume() so ASR can run while we still hear whether the user continues. Returns audio, "quit" or None (idle)."""
    pre, cur, speech, loud, quiet, t_idle, peak = [], [], False, 0, 0.0, time.time(), 0.0
    st = sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=1600, device=mic, latency="high"); st.start()
    keep = False
    try:
        while True:
            x, _ = st.read(1600); x = x[:, 0].copy(); rms = float(np.sqrt((x ** 2).mean()))
            if not FLOOR[0]: FLOOR[0] = rms
            thr = max(VAD_MIN, 3 * FLOOR[0])
            if select.select([KEY], [], [], 0)[0]:
                if KEY.readline().strip().lower() == "q": return "quit"
            if not speech:
                pre = (pre + [x])[-5:]  # 0.5 s pre-roll: the soft first syllable of the wake word ('Ki-') must not be clipped
                loud = loud + 1 if rms > thr else 0
                if loud >= 2:
                    speech, cur, quiet, peak = True, pre[:], 0.0, rms
                    if ONSET[0]: ONSET[0]()
                else:
                    if rms <= thr: FLOOR[0] = 0.95 * FLOOR[0] + 0.05 * rms  # adapt only on quiet blocks
                    if time.time() - t_idle > idle_s: return None
                print(f"\r  hört {'#' * min(30, int(rms / thr * 10)):<30} {'SPRICHT' if speech else '       '}", end="", flush=True)
                continue
            cur.append(x); peak = max(peak, rms)
            quiet = 0.0 if rms > thr else quiet + 0.1
            print(f"\r  hört {'#' * min(30, int(rms / thr * 10)):<30} SPRICHT", end="", flush=True)
            dur = len(cur) * 0.1
            if quiet >= silence_s or dur >= max_s:
                if dur - quiet < min_s:  # click/cough → keep waiting
                    speech, loud, cur = False, 0, []; continue
                y = np.concatenate(cur)
                LAST.clear(); LAST.update(t_end=time.perf_counter() - quiet, st=st, thr=thr); keep = True
                mic_log({"event": "mic", "mode": "vad", "dur_s": round(dur, 2), "speech": True, "stop": "silence" if quiet >= silence_s else "max", "hangover_ms": round(quiet * 1000),
                         "floor": round(FLOOR[0], 4), "thr": round(thr, 4), "peak_rms": round(peak, 4), "device": str(mic or "default")})
                return y
    finally:
        if not keep: st.stop(); st.close()
        print()


def make_watch():
    """open-mic watcher on the stream listen() left open (see watch.py); the engine restarts the turn if the user resumes before the first audio."""
    from watch import Watch
    st = LAST.get("st")
    if st is None: return None
    LAST["st"] = None
    return Watch(st, LAST.get("thr", 0.004), LAST.get("t_end", time.perf_counter()))


CONT_S = 2.5
R_stop = lambda q: __import__('rules').stop_word(q)


def mic_log(rec):  # every recording (also aborted ones) → same daily log as turns
    import datetime
    d = LOGDIR; d.mkdir(exist_ok=True); now = datetime.datetime.now()
    with open(d / f"{now:%Y-%m-%d}.jsonl", "a") as f: f.write(json.dumps({"ts": now.isoformat(timespec="seconds"), **rec}) + "\n")


def show(r):
    m = r["ms"]
    print(f"  du:      {r['transcript']}" + (f"   [{r['gate']}]" if r.get("gate") else ""))
    print(f"  tool:    {r.get('tool')} {json.dumps(r.get('args', {}), ensure_ascii=False)}")
    print(f"  kitsune: {r['answer']}")
    print("  ms: " + " ".join(f"{k[:-3]}={v}" for k, v in m.items() if k.endswith("_ms") and k != "e2e_feed_ms") + f"  peak={m['peak_gb']}GB audio={r['audio_s']}s" + (f" filler={m['filler']}" if "filler" in m else ""))


def handsfree(k, sd, E, mic, idle_s, no_barge=False, start="standby"):
    """The hands-free loop (also driven by sim/ with virtual devices). start: initial mode (standby = wake word 'Kitsune' needed; None = leave k.mode as is)."""
    from barge import BargeIn
    k.handsfree = True
    import tools as T; ONSET[0] = lambda: k.mode == "active" and T.prefetch_kal()
    if start: k.set_mode(start if k.standby_s else "active")
    x, from_barge, cont, t_prev = None, False, None, 0.0
    while True:
        if x is None: x = listen(sd, mic=mic, idle_s=idle_s)
        if x is None: print("Leerlauf, entlade Modelle."); return
        if isinstance(x, str): return
        t_end = LAST.get("t_end") if not from_barge else None
        t_a = time.perf_counter(); sink = E.new_sink(); t_b = time.perf_counter()
        bi = None if no_barge else BargeIn(sink, mic=mic, log=mic_log)
        if bi: bi.hit = k.cancel; bi.verify = None if os.environ.get("KITSUNE_BARGE_VERIFY", "1") == "0" else k.barge_verify; bi.start()  # echo-aware: you can interrupt while Kitsune speaks
        t_c = time.perf_counter(); watch = None if from_barge else make_watch()
        k.setup = {"sink_open_ms": round((t_b - t_a) * 1000), "barge_start_ms": round((t_c - t_b) * 1000), "watch_ms": round((time.perf_counter() - t_c) * 1000), "t_listen": t_a}  # pre-turn latency diagnostics (engine adds lag_ms)
        r = k.turn(x, sink, t_end=t_end, resume=watch, barge=from_barge, cont=cont); cont = None; sink.wait(); show(r); x, from_barge = None, False
        silent = r.get("tool") in ("standby_drop", "gate_drop")
        if not silent: k.touch(kit=r.get("audio_s", 0) > 0 and not (r.get("gate") or {}).get("relaxed"))
        if r.get("audio_s", 0) > 0 and r.get("tool") not in ("gate_hm", "wake", "barge_false"): k.t_ans = time.monotonic()  # last real answer (conversation-window anchor for the gate)
        if LAST.get("st"):  # turn did not call resume (e.g. text) → don't leak the stream
            LAST["st"].stop(); LAST["st"].close(); LAST["st"] = None
        if bi:
            bi.stop_ev.set(); bi.done.wait()
            if bi.audio is not None:
                f = sink.frac_played(); k.interrupted(f); bi.stats["frac_played"] = round(f, 2)
                print(f"  (unterbrochen nach {f:.0%} der Antwort, Reaktion {bi.stats.get('react_ms')} ms)"); x, from_barge = bi.audio, True
                if bi.stats.get("onset", 1e9) - (t_end or 0) < CONT_S and not R_stop(r["transcript"]): cont = r["transcript"]  # user kept talking right after the end of speech: same sentence
                continue
        if not silent: time.sleep(max(0.0, 0.3 - k.shadow_flush())); drain()  # let room echo of the answer die down (shadow d1 picks use that pause)
        elif k.shadow_q: k.shadow_flush()


def main():
    ap = argparse.ArgumentParser(prog="kitsune")
    ap.add_argument("--voice", default=os.environ.get("KITSUNE_VOICE", "female"), help="female|male|<full TTS system prompt>")
    ap.add_argument("--dry-run", action="store_true", help="kal add is not executed")
    ap.add_argument("--brain", default="gemma", help="gemma|lfm")
    ap.add_argument("--idle", type=int, default=IDLE_S)
    ap.add_argument("--mic", default=os.environ.get("KITSUNE_MIC", "MacBook Pro-Mikrofon"), help="input device index or name part (see --mics; default internal mic)")
    ap.add_argument("--mics", action="store_true", help="list audio devices and exit")
    ap.add_argument("--ptt", action="store_true", help="push-to-talk with Enter instead of hands-free")
    ap.add_argument("--aec", action="store_true", default=os.environ.get("KITSUNE_AEC", "0") == "1", help="Apple Voice Processing echo cancellation via aec/kitsune-aec (mic+speaker through one unit); falls back to plain audio if the helper fails")
    ap.add_argument("--no-barge", action="store_true", help="disable interrupting Kitsune while it speaks")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("ask", help="wav file(s) -> answer wav, no playback"); a.add_argument("--wav", nargs="+", required=True)
    a.add_argument("--play", action="store_true"); a.add_argument("--json", help="write records here")
    t = sub.add_parser("text", help="skip ASR"); t.add_argument("q", nargs="+"); t.add_argument("--play", action="store_true"); t.add_argument("--json")
    s = sub.add_parser("say"); s.add_argument("text"); s.add_argument("--out", default=str(OUT / "say.wav")); s.add_argument("--play", action="store_true")
    n = ap.parse_args()
    if n.mics:
        import sounddevice as sd; print(sd.query_devices()); return
    mic = int(n.mic) if n.mic and n.mic.isdigit() else n.mic

    import engine as E
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading models ...", file=sys.stderr, flush=True)
    k = E.Kitsune(voice=n.voice, dry=n.dry_run, brain=n.brain)
    print(f"ready: load {k.load_s}s warm {k.warm_s}s brain={k.brain_name}", file=sys.stderr, flush=True)
    from mlx_audio.audio_io import write

    def play(sink_play): return E.SpeakerSink() if sink_play else E.WavSink()

    if n.cmd == "say":
        sink = play(n.play); m = {}; t0 = time.perf_counter()
        k.speak(n.text, sink, m, t0); sink.wait()
        write(n.out, sink.audio(), samplerate=24000, format="wav"); print(n.out, m)
    elif n.cmd in ("ask", "text"):
        srcs = [("text", " ".join(n.q))] if n.cmd == "text" else n.wav
        recs = []
        for i, src in enumerate(srcs):
            name = pathlib.Path(src).stem if n.cmd == "ask" else f"text{i}"
            sink = play(n.play); r = k.turn(src, sink, barge=name.startswith("b")); sink.wait()  # b*.wav = captured after a barge-in
            r["wav"] = str(OUT / f"{name}_answer.wav"); r["src"] = str(src)
            write(r["wav"], sink.audio(), samplerate=24000, format="wav")
            show(r); recs.append(r)
        if n.json: pathlib.Path(n.json).write_text(json.dumps({"load_s": k.load_s, "warm_s": k.warm_s, "brain": k.brain_name, "turns": recs}, ensure_ascii=False, indent=1))
    else:
        import sounddevice as sd
        global KEY
        try:  # gpu-run starts us with "cmd &" → stdin is /dev/null; read the keyboard from the controlling terminal
            KEY = sys.stdin if sys.stdin.isatty() else open("/dev/tty")
        except OSError:
            sys.exit("kitsune: Mikrofon-Modus braucht ein Terminal (kein '!' in Claude Code, kein Hintergrundjob).")
        if not n.ptt:
            print(f"Freihändig: im Standby reagiert Kitsune nur auf den Namen ('Kitsune, ...'), danach Gespräch ohne Namen; {k.standby_s:.0f} s Pause oder 'Kitsune, Pause' -> Standby. 0.4 s Pause = fertig. q + Enter = beenden. Leerlauf {n.idle}s -> Entladen.", flush=True)
            if n.aec:
                import aec; sd = aec.start(log=lambda s: print(s, flush=True)) or sd   # None -> plain sounddevice path
            try: return handsfree(k, sd, E, mic, n.idle, n.no_barge)
            finally:
                if n.aec: aec.stop()
        print("Enter = sprechen, Enter = fertig (oder 0.8 s Stille). q + Enter = beenden. Leerlauf "
              f"{n.idle}s -> Entladen.", flush=True)
        last = time.time()
        while True:
            print("> ", end="", flush=True)
            r_, _, _ = select.select([KEY], [], [], 1.0)
            if not r_:
                if time.time() - last > n.idle: print("\nLeerlauf, entlade Modelle."); return
                continue
            line = KEY.readline()
            if not line or line.strip().lower() == "q": return  # EOF or quit
            drain(); print("  (höre zu ...)", flush=True)
            x = record(sd, mic=mic)
            if len(x) < 8000: print("  zu kurz (Enter erst nach dem Sprechen drücken)"); continue
            sink = E.new_sink()  # mic is closed during playback = half-duplex
            r = k.turn(x, sink); sink.wait(); show(r); last = time.time(); k.shadow_flush()


if __name__ == "__main__":
    main()
