"""kitsuned core: LFM3-Audio (ASR+TTS+router) + Gemma brain in one process."""
import sys, os, re, json, time, pathlib, tempfile, dataclasses, copy, hashlib, concurrent.futures
from datetime import datetime

HOME = pathlib.Path.home()
sys.path.insert(0, str(__import__('os').environ.get('KITSUNE_MLXVLM_SRC') or HOME / 'Claude-Workspace/lfm3/mlx-vlm'))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import mlx.core as mx
mx.set_cache_limit(int(float(os.environ.get('KITSUNE_MLX_CACHE_GB', '2')) * 2**30)); mx.set_memory_limit(int(float(os.environ.get('KITSUNE_MLX_MEM_GB', '24')) * 2**30))  # uncapped MLX once reached 50 GB
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.generate import PromptCacheState
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs, should_add_special_tokens
import mlx_vlm.structured as st
import llguidance as llg, llguidance.hf
from mlx_audio.audio_io import write as wav_write

import tools as T, dates as D, rules as R, ctx as C, lexicon as LX
from codec_stream import StreamingCodec

LIQ = HOME / "AI-Model/models/LiquidAI"
GEMMA = HOME / ".lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit"
VOICES = {"female": "Perform TTS in German using a female voice.", "male": "Perform TTS in German using a male voice."}
SPEED = {"fast": ", speaking quickly at a fast pace.", "slow": ". Speak slowly.", "normal": "."}  # prompt style; KITSUNE_SPEED (2026-11-06 listening test: fast liked)
TEMPO_F = pathlib.Path(__file__).parent / "cache" / "tempo.json"  # spoken 'schneller/langsamer' persists here
LOGS = pathlib.Path(__file__).parent / "logs"
SENT = re.compile(r"(.+?(?:[!?…]|(?<!\b\d)(?<!\b\d\d)(?<!\b\w)(?<!\bca)(?<!\bNr)(?<!\bArt)(?<!\bAbs)(?<!\bbzw)(?<!\busw)(?<!\bMio)(?<!\bMrd)(?<!\bSt)(?<!\bDr)(?<!\bProf)\.))(\s+|$)", re.S)  # no split after '2.' / 'z.' / 'ca.': 'am 2. August' stays one sentence for speakable
CLAUSE = re.compile(r"((?:\S+\s+){%d,}?\S+?[,;:–])(\s+)" % (int(os.environ.get("KITSUNE_CLAUSE_WORDS", "3")) - 1), re.S)  # first speakable clause: >= N words ending in , ; : –
clean = lambda s: C.speakable(re.sub(r"[*_#`]+|\([^)]*\)", "", s).replace("  ", " ").strip())
ms = lambda a, b: round((b - a) * 1000)

BRAIN_SYS = C.SELF + " " + C.PROFILE + " " + C.ASR + " " + C.STYLE
BRAIN_TOOL = (" Antworte AUSSCHLIESSLICH mit Informationen aus dem Werkzeugergebnis. Steht die Antwort dort nicht oder passt das Ergebnis nicht zur Frage, "
              "sage: Das weiß ich nicht genau. Erfinde nichts.")
BRAIN_OWN = " Antworte aus deinem Wissen. Bist du unsicher, sag es ehrlich."
BRAIN_SUCHE = (" Antworte mit konkreten Fakten aus den Treffern (Zahlen, Namen, Ergebnisse). Enthalten die Treffer die Antwort nicht, sage in EINEM Satz, dass sie das nicht hergeben. "
               "Beschreibe niemals die Suchergebnisse oder Webseiten und nenne keine Webseiten.")
BRAIN_BUCH = (" Antworte in höchstens drei kurzen Sätzen. Nenne die Quelle gesprochen als 'laut <Autor>, <Werk>', ohne Seitenzahlen und ohne Adressen. "
              "Steht die Antwort nicht in den Treffern, sage: Das steht nicht in der Bibliothek.")
PERSONA = (C.SELF + " " + C.PROFILE + " " + C.STYLE + " Wiederhole niemals den Satz des Nutzers. Behaupte keine Fähigkeiten, die du nicht hast. "
           "Bei Beleidigungen oder anzüglichen Bemerkungen: kurz, ruhig und freundlich abwinken, keine Moralpredigt, nicht einsteigen.")
FILL_TXT = {"web": "Moment, ich schau nach.", "lies": "Einen Moment, ich lese das.", "kurz": "Einen Augenblick.", "eintragen": "Ich trage das ein.", "br1": "Moment noch.", "br2": "Gleich.", "br3": "Ja, gleich."}
BRIDGE = ["br1", "br2", "br3"]
CHAIN = int(os.environ.get("KITSUNE_FILL_CHAIN", "2"))  # max chained bridge fillers while the answer audio is still missing after a filler (0 = off)
LEAD_TXT = {"gute_frage": "Gute Frage.", "also": "Also,", "mal_sehen": "Mal sehen,", "kurz": "Kurz gesagt,", "gut_wissen": "Soweit ich weiß,", "hm_ja": "Hm, ja,"}
LEAD = {"warum": ["gute_frage", "also", "mal_sehen"], "wer": ["gut_wissen", "also"], "wie": ["also", "mal_sehen", "kurz"], "default": ["also", "mal_sehen", "hm_ja", "kurz"]}
FILL = {"netz_suche": ("web", 0), "netz_news": ("web", 0), "wetter": ("web", .2), "netz_lies": ("lies", 0),
        "wiki_suche": ("kurz", .15), "bibliothek": ("kurz", .15), "kal_heute": ("kurz", .15), "kal_woche": ("kurz", .15), "fakt": ("kurz", .15), "gpu_status": ("kurz", .15)}
# the name hint raised the wake-word recall from 6/10 to 9/10 (Thorsten voice in the office room); echo of the hint on noise is filtered
ASR_SYS = "Perform ASR in German." + (" The assistant is called Kitsune." if os.environ.get("KITSUNE_ASR_HINT", "1") != "0" else "")
ASR_ECHO = re.compile(r"assistant|is called|assistent ist|perform asr", re.I)
HANGOVER = 0.4  # full end-of-speech hangover; routing may start earlier (listen() hands over after ~0.25-0.3 s of silence, speculative)
LEAD_ON = os.environ.get("KITSUNE_LEAD", "0") == "1"  # Gemma lead-ins ('Also,' 'Gute Frage.') are off: the audible gap between lead-in and answer was worse than the saved latency
BRAIN_CACHE = os.environ.get("KITSUNE_BRAIN_CACHE", "1") != "0"
PIPE = os.environ.get("KITSUNE_PIPE", "0") == "1"
FASTROUTE = os.environ.get("KITSUNE_FASTROUTE", "1") != "0"
ROUTER = os.environ.get("KITSUNE_ROUTER", "shadow")  # lfm | d1 (d1 picks the tool, args by rule, arg-heavy/unsure -> lfm) | shadow (lfm decides, d1 pick only logged)
BARGE_Q = os.environ.get("KITSUNE_BARGE_Q", "barge")  # barge: d1 'will der Nutzer unterbrechen?' with Kitsune's current sentence | addr: old addressee question
BARGE_P, BARGE_PQ = float(os.environ.get("KITSUNE_BARGE_P", "0.45")), float(os.environ.get("KITSUNE_BARGE_PQ", "0.25"))
D1_CONF = float(os.environ.get("KITSUNE_D1_CONF", "0.6"))
D1_TOOLS = [(t["function"]["name"], t["function"]["description"]) for t in T.TOOLS]
D1_FRAGE = {"wissen", "wiki_suche", "bibliothek", "fakt", "netz_suche"}
D1_NOARG = {"kal_heute", "kal_woche", "gpu_status", "plaudern", "netz_news"}
CHAT_DEMOTE = os.environ.get("KITSUNE_CHAT_DEMOTE", "1") != "0"  # rules.chat_demote: statements / self questions / calendar pick without calendar cue -> plaudern|wissen
TALKER = os.environ.get("KITSUNE_TALKER", "gemma")  # opt-in experiment: 'lfm' = LFM3-Audio writes small talk (talk.py); default unchanged
if TALKER == "lfm": import talk as TK
PRE = [0.25]  # sink pre-buffer s; raised to 0.35 after any underrun/xrun (adaptive)
yes_no = R.yes_no


class Restart(Exception):
    """user kept talking before the first audio sample was fed: redo the turn on the concatenated audio."""


class WavSink:
    def __init__(self): self.parts, self.t_first, self.skip, self.t_play = [], None, 0, None  # skip: filler samples (not part of the answer)
    def feed(self, pcm):
        if self.t_first is None and len(pcm): self.t_first = time.perf_counter()
        self.parts.append(np.asarray(pcm, np.float32))
    def audio(self): return np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)
    def wait(self): pass


class SpeakerSink(WavSink):
    """sounddevice 24 kHz output; callback drains a buffer, pre-buffers PRE s (adaptive 0.25 → 0.35 after dropouts), silence on underrun."""
    def __init__(self, pre=None):  # 0.15 s ran dry on slow first TTS frames → stutter
        super().__init__(); pre = PRE[0] if pre is None else pre
        import sounddevice as sd, threading
        self.buf, self.lock, self.pre, self.playing, self.under, self.done = np.zeros(0, np.float32), threading.Lock(), int(pre * 24000), False, 0, False
        self.paused = False  # barge-in candidate under verification: output silence, keep the buffer
        self.log, self.fed, self.out = [], 0, 0  # log: (monotonic time the block hits the speaker, samples) for barge-in reference
        # fixed 1024-frame (43 ms) blocks + high latency: tiny default blocks starved the Python callback under load → stutter
        self.st = sd.OutputStream(samplerate=24000, channels=1, dtype="float32", callback=self.cb, blocksize=1024, latency="high")
        self.xruns = 0
        self.st.start()
    def cb(self, out, frames, ti, status):
        if status.output_underflow: self.xruns += 1  # CoreAudio-level dropout (late callback)
        with self.lock:
            if not self.playing and (len(self.buf) >= self.pre or self.done): self.playing = True; self.t_play = time.perf_counter() + self.st.latency
            n = 0 if self.paused else (min(frames, len(self.buf)) if self.playing else 0)
            out[:n, 0] = self.buf[:n]; out[n:, 0] = 0
            self.buf = self.buf[n:]; self.out += n
            if n:
                t = time.monotonic() + self.st.latency
                self.log = [e for e in self.log if e[0] > t - 2.0] + [(t, out[:n, 0].copy())]  # last 2 s by time, not by count
            if self.playing and n < frames and not self.done and not self.paused: self.under += 1
    def feed(self, pcm):
        if self.done: return  # stopped by barge-in
        super().feed(pcm); self.fed += len(pcm)
        with self.lock: self.buf = np.concatenate([self.buf, np.asarray(pcm, np.float32)])
    def played_between(self, t0, t1):
        with self.lock: blocks = [b for t, b in self.log if t0 <= t <= t1]
        return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
    def pause(self):
        with self.lock: self.paused = True
    def resume(self):
        with self.lock: self.paused = False
    def stop_now(self):  # barge-in: drop everything not yet played
        with self.lock: self.buf = np.zeros(0, np.float32); self.done = True; self.paused = False
    def frac_played(self): return max(0, self.out - self.skip) / (self.fed - self.skip) if self.fed > self.skip else 1.0
    def wait(self):
        self.done = True
        while True:
            with self.lock:
                if len(self.buf) == 0: break
            time.sleep(0.03)
        time.sleep(0.15); self.st.stop(); self.st.close()


_AEC_SINK = []


def new_sink():
    """output sink: the AEC helper's sink while `--aec` is up, else the sounddevice SpeakerSink (also the fallback when the helper died)."""
    try:
        import aec
        if aec.ok():
            if not _AEC_SINK: _AEC_SINK.append(aec.make_sink_cls(WavSink))
            return _AEC_SINK[0]()
    except Exception as e: print("aec sink off:", e, file=sys.stderr)
    return SpeakerSink()


class Kitsune:
    def __init__(self, voice="female", n_hist=6, dry=False, brain="gemma"):
        self.voice = VOICES.get(voice, voice); self.n_hist = n_hist; self.dry = dry
        sp = os.environ.get("KITSUNE_SPEED", "fast")
        if voice in VOICES and sp in SPEED: self.voice = self.voice.rstrip(".") + SPEED[sp]
        try: self.tempo = float(os.environ.get("KITSUNE_TEMPO") or json.loads(TEMPO_F.read_text())["tempo"])
        except Exception: self.tempo = 1.15
        self._fst = {}
        os.environ["KITSUNE_DRYRUN"] = "1" if dry else os.environ.get("KITSUNE_DRYRUN", "0")
        t = time.perf_counter()
        self.A, self.Ap = load(str(LIQ / "LFM3-Audio-2B-MLX-8bit"))
        self.brain_name = brain
        try:
            if brain != "gemma": raise RuntimeError("fallback requested")
            self.B, self.Bp = load(str(GEMMA)); self.brain_name = "gemma-4-E4B-4bit"
        except Exception as e:
            print("brain fallback LFM3-1.7B:", e, file=sys.stderr)
            self.B, self.Bp = load(str(LIQ / "LFM3-1.7B-MLX-8bit")); self.brain_name = "LFM3-1.7B-8bit"
        tk = getattr(self.Ap, "tokenizer", self.Ap)
        self.ltok = llg.hf.from_tokenizer(tk)
        ids = {k: tk.convert_tokens_to_ids(k) for k in T.TOOL_IDS}
        self.grammar = llg.grammar_from("lark", T.lark(ids))
        self.sc = StreamingCodec(self.A.codec())
        self.hist, self.pending, self.last_items, self.prev_user = [], None, [], ""
        self.topics, self.last_place, self.foreign, self.last_lead, self.clean_turns, self.gpu_t, self.gpu_free_ = [], "", None, "", 0, 0.0, False
        self.ignoring, self.rc, self.fillers, self.route_info, self.bc = False, None, {}, {}, {}
        self.pool = concurrent.futures.ThreadPoolExecutor(2)
        import threading; self._fl, self._pe, self.ans_started, self._bth = threading.Lock(), 0.0, False, None
        # conversation state: standby (wake word needed) <-> active (conversation mode)
        self.handsfree, self.mode = False, "active"
        self.standby_s = float(os.environ.get("KITSUNE_STANDBY_S", "45"))  # 0 = never go back to standby
        self.t_act, self.t_kit, self.rejects, self.bg_until = time.monotonic(), -1e9, [], 0.0
        self.watch, self.t_end0, self.pend_prev = None, 0.0, None
        self.gate_ans, self.gate_hm, self.gate_bg = (float(os.environ.get(k, d)) for k, d in (("KITSUNE_GATE_ANS", "0.8"), ("KITSUNE_GATE_HM", "0.6"), ("KITSUNE_GATE_BG", "0.9")))
        self.addr = self.d1 = None; self.d1_res = None; self.shadow_q = []
        gate_on = os.environ.get("KITSUNE_GATE", "1") != "0"
        if gate_on or ROUTER != "lfm":
            try:
                from addressee import Addressee
                self.d1 = Addressee(); self.addr = self.d1 if gate_on else None
            except Exception as e: print("d1 off (gate/router):", e, file=sys.stderr)
        import threading; self.cancel = threading.Event()  # set by barge-in
        self.load_s = round(time.perf_counter() - t, 1)
        self.warm()
        self.load_fillers()

    # ---- warmup ----
    def warm(self):
        t = time.perf_counter()
        self.asr(np.zeros(48000, np.float32))
        self.route("Hallo", datetime.now(), warm=True); self.route("Wie wird das Wetter?", datetime.now(), warm=True)
        "".join(self.brain_stream([{"role": "user", "content": "Sag Hallo."}], 8))
        for _ in self.tts_stream("Hallo."): pass
        self.warm_s = round(time.perf_counter() - t, 1)

    # ---- ASR ----
    def asr(self, wav):
        tmp = None
        if not isinstance(wav, (str, pathlib.Path)):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
            wav_write(tmp.name, np.asarray(wav, np.float32), samplerate=16000, format="wav"); wav = tmp.name
        p = apply_chat_template(self.Ap, self.A.config, [{"role": "system", "content": ASR_SYS}, {"role": "user", "content": ""}], num_audios=1)
        try:
            t = generate(self.A, self.Ap, p, audio=[str(wav)], temperature=0, max_tokens=256).text.strip()
            return "" if ASR_ECHO.search(t) else t  # on noise the hint can be echoed ('The assistant is called Kitsune.') -> treat as empty
        finally:
            if tmp: os.unlink(tmp.name)

    # ---- router (grammar e) ----
    def _prefix_cache(self, pref):
        """KV state after the static system+tools prefix (+ '<|im_start|>' forced, so cache offset == cached token count; hybrid conv layers can't be trimmed)."""
        if self.rc and self.rc[0] == pref: return self.rc[1]
        tk = getattr(self.Ap, "tokenizer", self.Ap); ims = tk.convert_tokens_to_ids("<|im_start|>")
        force = lambda tokens, logits: mx.where(mx.arange(logits.shape[-1]) == ims, logits, -1e9)
        s0 = PromptCacheState()
        for _ in stream_generate(self.A, self.Ap, pref, max_tokens=1, temperature=0.0, prompt_cache_state=s0, logits_processors=[force]): pass
        self.rc = (pref, s0); return s0

    def route(self, utter, now, warm=False):
        if len(R.W(utter)) >= 5: utter = re.sub(r"^(und|aber|also|dann|okay|ok)[,\s]+(?=(wie|was|wer|wo|wann|warum|wieso)\b)", "", utter, flags=re.I)  # 'Und wie viele ...' is a new question, not a follow-up
        msgs = [{"role": "system", "content": T.sys_router()}]  # static → byte-identical prefix
        for u, a in self.hist[-2:]:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a[:200]}]
        msgs.append({"role": "user", "content": C.state_line(now, C.ort(self.last_place), self.mode, self.pending) + utter})
        p = apply_chat_template(self.Ap, self.A.config, msgs, tools=T.TOOLS, enable_thinking=False)
        state = None
        try:
            s0 = self._prefix_cache(p[:p.index("<|im_start|>user")])
            state = PromptCacheState(); state.cache = copy.deepcopy(s0.cache); state.token_ids = list(s0.token_ids)
        except Exception as e: print("route cache off:", e, file=sys.stderr)
        proc = st.LLGuidanceLogitsProcessor(self.grammar, self.ltok)  # fresh per call
        t = time.perf_counter(); tf, txt, last = None, "", None
        for r in stream_generate(self.A, self.Ap, p, max_tokens=160, temperature=0.0, logits_processors=[proc], prompt_cache_state=state):
            if tf is None: tf = time.perf_counter()
            txt += r.text; last = r
        te = time.perf_counter()
        self.route_info = {"route_prefill_ms": ms(t, tf or te), "route_decode_ms": ms(tf or te, te), "route_tokens": getattr(last, "prompt_tokens", None), "route_cached": getattr(last, "cached_tokens", None)}
        return T.parse_call(txt)

    def d1_args(self, tool, conf, q):
        """d1 tool pick -> (name, args) when the args follow from rules; None -> LFM3 router (arg-heavy tools, low confidence, follow-ups that need the history)."""
        if conf < D1_CONF: return None
        if tool in D1_NOARG: return tool, {}
        if tool == "antwort": return tool, {"text": "Ja?"}
        if tool in D1_FRAGE:
            if self.hist and (len(R.W(q)) <= 4 or re.search(r"\b(er|sie|ihm|ihr|ihn|dort|da|davon|darüber|dazu|dem|denen|deren|dessen)\b", q.lower())): return None  # needs a standalone rewrite from the history
            return tool, {"frage": q}
        if tool == "wetter" and (sl := R.wx_slots(q)) is not None: return tool, sl
        return None

    def route_final(self, q, now, m=None):
        """fast_route / router + post-guards -> (name, args). m: optional metrics dict."""
        m = {} if m is None else m
        if FASTROUTE and (fr := R.fast_route(q)): name, args = fr; m["fast_route"] = True
        else:
            name = None
            if ROUTER == "shadow" and self.d1 is not None and not self.d1_res: m["d1_later"] = True  # shadow without a free gate pass: d1 runs after the answer (shadow_flush), no added latency
            elif ROUTER != "lfm" and self.d1 is not None:
                r = self.d1_res or self.d1.ask(q, self.hist[-2:], D1_TOOLS, addr=False); self.d1_res = None
                m.update(d1_tool=r["tool"], d1_conf=r["conf"], d1_ms=0 if "p" in r else r["ms"])
                if ROUTER == "d1" and (fa := self.d1_args(r["tool"], r["conf"], q)): name, args = fa; m["router"] = "d1"
            if name is None: name, args = self.route(q, now); m.update(self.route_info)
        if name is None: name, args = "wissen", {"frage": q}
        if TALKER == "lfm" and name == "plaudern" and "2" in TK.LAYERS and TK.fact_q(q): name, args = "wissen", {"frage": q}  # opt-in talker experiment: factual questions go to Gemma
        if name == "antwort" and (R.echo(args.get("text", ""), q) or len(R.W(args.get("text", ""))) > 5): name, args = "plaudern", {}  # never parrot the user
        if name == "antwort":  # canned acks by keyword (router text is unreliable)
            lq = q.lower(); args = {"text": "Gern geschehen." if "danke" in lq else "Tschüss!" if re.search(r"tschüss|bis bald|auf wiedersehen", lq) else args.get("text", "Ja?")}
        if name == "wetter" and not R.WX.search(q) and not (self.pending or {}).get("stage") == "ort": name, args = "wiki_suche", {"frage": q}  # router drift guard
        if name == "wetter" and (o := args.get("ort", "")) and not R.grounded(o, q): m["ort_dropped"] = o; args = {**args, "ort": ""}  # router invented a place ('[Ortsinformationen werden benötigt]') -> default place / ask
        if CHAT_DEMOTE and not m.get("fast_route") and (dm := R.chat_demote(name, q, bool(self.pending))): m["demoted"] = name; name, args = dm, ({"frage": q} if dm == "wissen" else {})  # no clear tool intent -> small talk (Gemma persona)
        if m.pop("d1_later", None): self.shadow_q = (self.shadow_q + [(q, list(self.hist[-2:]), name, datetime.now())])[-20:]
        return name, args

    def set_tempo(self, cmd):
        """'faster' / 'slower' / 'normal' -> new playback tempo (0.8 .. 1.5, steps of 0.1), persisted; returns the spoken confirmation."""
        t = {"faster": self.tempo + 0.1, "slower": self.tempo - 0.1, "normal": 1.15}[cmd]
        self.tempo = round(min(1.5, max(0.8, t)), 2)
        try: TEMPO_F.parent.mkdir(exist_ok=True); TEMPO_F.write_text(json.dumps({"tempo": self.tempo}))
        except Exception: pass
        return "Okay, so?" if cmd != "normal" else "Okay, wieder normal."

    def shadow_flush(self, n=1):
        """shadow router: d1 picks for turns routed without a free gate pass, computed once the answer has finished playing -> extra log lines {'shadow': ...}."""
        t = time.perf_counter()
        while self.shadow_q and self.d1 is not None and n > 0:
            n -= 1
            q, hist, lfm, ts = self.shadow_q.pop(0)
            try: r = self.d1.ask(q, hist, D1_TOOLS, addr=False)
            except Exception as e: print("shadow d1:", e, file=sys.stderr); continue
            LOGS.mkdir(exist_ok=True)
            with open(LOGS / f"{ts:%Y-%m-%d}.jsonl", "a") as f:
                f.write(json.dumps({"ts": ts.isoformat(timespec="seconds"), "shadow": {"q": q, "lfm": lfm, "d1_tool": r["tool"], "d1_conf": r["conf"], "d1_ms": r["ms"], "agree": r["tool"] == lfm}}, ensure_ascii=False) + "\n")
        return round((time.perf_counter() - t), 3)

    # ---- brain ----
    def brain_stream(self, msgs, max_tokens=160, temp=0.3, cache=True):
        tk = getattr(self.Bp, "tokenizer", self.Bp)
        p = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        kw = {}
        if cache and BRAIN_CACHE and (st_ := self._brain_prefix(p, msgs[0]["content"])) is not None: kw["prompt_cache_state"] = st_
        acc = ""
        for r in stream_generate(self.B, self.Bp, p, max_tokens=max_tokens, temperature=temp, top_p=0.95, **kw):
            acc += r.text
            if "</think>" in acc: acc = acc.split("</think>")[-1]; continue
            yield r.text

    def _brain_prefix(self, p, sysc):
        """KV state after the static system prompt (the first line block of the system message, before the per-turn state/summary lines); copied per call."""
        try:
            stat = sysc.split("\n", 1)[0]; i = p.index(stat) + len(stat)
            pref = p[:i]; hit = self.bc.get(pref)
            if hit is None:
                s0 = PromptCacheState()
                for _ in stream_generate(self.B, self.Bp, pref, max_tokens=1, temperature=0.0, prompt_cache_state=s0): pass
                n = max(c.offset for c in s0.cache if hasattr(c, "offset")); s0.token_ids = s0.token_ids[:n]  # the sampled token's KV is not in the cache
                hit = self.bc[pref] = s0
            state = PromptCacheState(); state.cache = copy.deepcopy(hit.cache); state.token_ids = list(hit.token_ids); return state
        except Exception as e:
            print("brain cache off:", e, file=sys.stderr); return None

    def brain_msgs(self, q, tool_out=None, name=None):
        sysm = PERSONA if name == "plaudern" else BRAIN_SYS + (BRAIN_TOOL + (BRAIN_SUCHE if name == "netz_suche" else BRAIN_BUCH if name == "bibliothek" else "") if tool_out is not None else BRAIN_OWN)
        extra = "\n".join(x for x in (C.state_line(datetime.now(), C.ort(self.last_place)).strip("[] "), C.summary(self.topics, self.n_hist), C.handle(self.last_items)) if x)
        msgs = [{"role": "system", "content": sysm + "\n" + extra}]
        for u, a in self.hist[-self.n_hist:]: msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": q if tool_out is None else f"Frage: {q}\n\nErgebnis von {name}:\n{T.trim_ctx(name, tool_out)}"})
        return msgs

    # ---- TTS with streaming codec ----
    def tts_stream(self, text, first=1, rest=2, lang=None):
        A, cfg = self.A, self.A.config
        prompt = apply_chat_template(self.Ap, cfg, [{"role": "system", "content": self.voice if not lang else self.voice.replace("in German", "in " + lang)}, {"role": "user", "content": text}])
        inp = prepare_inputs(self.Ap, images=None, audio=None, videos=None, prompts=prompt, add_special_tokens=should_add_special_tokens(cfg.model_type, self.Ap))
        ids = inp.pop("input_ids"); inp.pop("attention_mask", None)
        self.sc.reset(); buf, n = [], first
        for tok, codes in A.stream_slots(ids, **A.audio_sampling, **inp):
            if codes is None: continue
            if codes[0].item() == cfg.audio_eos_code: break
            buf.append(codes)
            if len(buf) >= n:
                yield np.asarray(self.sc.decode(mx.stack(buf, axis=-1))); buf, n = [], rest
        yield np.asarray(self.sc.decode(mx.stack(buf, axis=-1) if buf else None, finalize=True))

    def speak(self, text, sink, m, t0, lang=None):
        """lang: speak a foreign word with an explicit TTS language prompt ('Perform TTS in Bulgarian ...')."""
        if not (text := clean(text) if not lang else text): return
        self.said = [x for x in getattr(self, "said", []) if time.monotonic() - x[0] < 90][-12:] + [(time.monotonic(), text)]  # recent own speech: barge_verify vetoes echoes of it
        import tempo as TP
        st = TP.Stretch(getattr(self, "tempo", 1.0))
        import itertools
        for pcm in itertools.chain(self.tts_stream(text, lang=lang), [None]):  # lazy: stream chunk by chunk (a list here waited for the whole sentence, +1 s)
            if self.cancel.is_set(): break
            pcm = st.push(np.zeros(0, np.float32), final=True) if pcm is None else st.push(pcm)
            if len(pcm):
                with self._fl:  # the bridge thread must not slip a filler in between
                    self._commit(); self.ans_started = True; sink.feed(pcm); self._note(len(pcm))
                m.setdefault("first_audio_ms", ms(t0, time.perf_counter())); m.setdefault("speak_first_ms", ms(t0, time.perf_counter()))  # first ANSWER sample (fillers excluded)


    class _Inline:
        err = None
        def __init__(self, k, sink, m, t0): self.a = (sink, m, t0); self.k = k
        def put(self, t): self.k.speak(t, *self.a)
        def close(self): pass

    class _Pipe:
        def __init__(self, k, sink, m, t0):
            import queue, threading
            self.q, self.err, self.k, self.a = queue.Queue(), None, k, (sink, m, t0)
            self.th = threading.Thread(target=self.run, daemon=True); self.th.start()
        def run(self):
            while (t := self.q.get()) is not None:
                if self.err: continue
                try: self.k.speak(t, *self.a)
                except BaseException as e: self.err = e
        def put(self, t): self.q.put(t)
        def close(self):
            self.q.put(None); self.th.join()
            if self.err: raise self.err

    # ---- filler audio (pre-rendered, same voice) ----
    def load_fillers(self):
        f = pathlib.Path(__file__).parent / "cache" / f"fillers-{hashlib.md5((self.voice + json.dumps(FILL_TXT) + json.dumps(LEAD_TXT)).encode()).hexdigest()[:10]}.npz"
        if f.exists():
            z = np.load(f); self.fillers = {k: z[k] for k in z.files}; return
        for k, txt in {**FILL_TXT, **LEAD_TXT}.items(): self.fillers[k] = np.concatenate([p for p in self.tts_stream(txt) if len(p)]).astype(np.float32)
        f.parent.mkdir(exist_ok=True); np.savez(f, **self.fillers)

    def lead(self, q, sink, m, t0):
        """short pre-rendered lead-in before a Gemma answer (first audio ~ end of routing); varied by question type, never the same twice in a row."""
        if not LEAD_ON: return
        w = R.W(q)[:1]; kind = "warum" if w and w[0] in ("warum", "wieso", "weshalb") else "wer" if w and w[0] in ("wer", "wem", "wen") else "wie" if w and w[0] in ("wie", "was", "wann", "wo") else "default"
        opts = [k for k in LEAD[kind] if k != self.last_lead and k in self.fillers] or [k for k in LEAD["default"] if k != self.last_lead and k in self.fillers]
        if not opts: return
        self.last_lead = opts[int(time.time() * 1000) % len(opts)]; self.filler(self.last_lead, sink, m, t0); m["lead"] = self.last_lead

    def _note(self, n):
        """estimated wall time at which everything fed so far has been played (sink-agnostic playhead)."""
        self._pe = max(self._pe, time.monotonic()) + n / 24000

    def filler(self, key, sink, m, t0, bridge=False):
        a = self.fillers.get(key)
        if a is None or self.cancel.is_set(): return
        if (tp := getattr(self, "tempo", 1.0)) != 1:
            fs = self.__dict__.setdefault("_fst", {})
            if (key, tp) not in fs: import tempo as TP; fs[(key, tp)] = TP.stretch(a, tp)
            a = fs[(key, tp)]
        self._commit(); sink.feed(a); sink.skip += len(a); self._note(len(a))
        if bridge: m["bridges"] = m.get("bridges", 0) + 1
        else: m["filler"] = key
        self.said = [x for x in getattr(self, "said", []) if time.monotonic() - x[0] < 90][-12:] + [(time.monotonic(), FILL_TXT.get(key) or LEAD_TXT.get(key, ""))]
        if not bridge: m.setdefault("first_audio_ms", ms(t0, time.perf_counter()))

    def bridge_start(self, sink, m, t0):
        """after a filler: while no answer audio has been fed and the playhead is about to run dry, chain another short pre-rendered filler (seamless, max CHAIN)."""
        if CHAIN <= 0 or self._bth is not None: return
        import threading
        self._bstop = threading.Event()
        def run():
            n, t_end = 0, time.monotonic() + 8
            while not self._bstop.is_set() and n < min(CHAIN, len(BRIDGE)) and time.monotonic() < t_end:
                time.sleep(0.02)
                with self._fl:
                    if self.ans_started or self.cancel.is_set(): return
                    if self._pe - time.monotonic() < 0.1:
                        try: self.filler(BRIDGE[n], sink, m, t0, bridge=True)
                        except Exception: return
                        n += 1
        self._bth = threading.Thread(target=run, daemon=True); self._bth.start()

    def bridge_end(self):
        if self._bth is not None: self._bstop.set(); self._bth.join(1); self._bth = None

    # ---- standby / addressee gate ----
    def touch(self, kit=False):
        """user interaction (or Kitsune audio end) -> restarts the standby timer / conversation window."""
        t = time.monotonic(); self.t_act = t
        if kit: self.t_kit = t

    def set_mode(self, mode, t_act=None):
        self.mode = mode
        if mode == "standby": self.pending, self.ignoring = None, False
        self.t_act = time.monotonic() if t_act is None else t_act

    def gate(self, q, pend, barge):
        """-> (verdict, info). verdict: pass | drop | hm | wake | end. Only in hands-free mode; ASR text q already available."""
        info = {}
        if not self.handsfree: return "pass", info
        t = time.monotonic(); named = R.NAME.search(q)
        if self.mode == "active" and self.standby_s and t - self.t_act > self.standby_s: self.set_mode("standby"); info["timeout"] = True
        if self.mode == "standby":
            if not named: return "drop", {**info, "why": "standby"}
            if R.ends_conversation(q): self.set_mode("standby"); return "end", info
            self.set_mode("active"); info["woke"] = True
            return ("wake" if R.only_name(q) else "pass"), info
        if R.stop_word(q) and not barge and t - self.t_kit > 2.0 and not pend: return "drop", {**info, "why": "nothing to stop"}  # 'Stopp.' while Kitsune is silent
        if R.ends_conversation(q): return "end", info
        bg = t < self.bg_until
        t_ans = getattr(self, "t_ans", -1e9)
        if not bg and t - t_ans <= 25 and R.is_continuation(q): return "pass", {**info, "why": "continuation"}  # 'und zwar K I M I C' right after an exchange: a follow-up, not TV
        in_win = barge or (t - self.t_kit <= (15 if pend else 8) and not bg)  # a barge-in capture is by definition inside the conversation window; only long declaratives reach the addressee model (headphones: TV voice)
        h = R.gate_hard(q, in_window=in_win, pending=bool(pend), ignoring=self.ignoring)
        if h == "pass": return "pass", {**info, "why": "named" if named else "hard"}
        if h == "drop": self.reject(t); return "drop", {**info, "why": "hard"}
        if self.addr is None: return "pass", info
        r = self.addr.ask(q, self.hist[-2:], D1_TOOLS if ROUTER != "lfm" else None); p = r["p"]; info.update(p=round(p, 3), gate_ms=self.addr.ms)
        if "tool" in r: self.d1_res = r  # same forward pass already picked a tool: route_final reuses it
        if p < 0.9 and not R.is_question_or_cmd(q) and not re.search(r"\b(du|dir|dich|bitte)\b", q.lower()): p *= 0.8 if len(R.W(q)) >= 6 else 0.9  # long undirected declaratives = TV/talk
        ans = self.gate_bg if bg else self.gate_ans
        if p >= ans: return "pass", info
        recent = t - t_ans <= 30 and t - getattr(self, "t_hm", -1e9) > 20 and len(R.W(q)) <= 8  # right after a real answer a middling p earns a short 'Hm?' (once per 20 s, short utterances, never in the TV/background profile)
        if p >= self.gate_hm and not bg and R.starts_directed(q): return "hm", info
        if p >= self.gate_hm and not bg and recent: self.t_hm = t; return "hm", {**info, "relaxed": True}  # relaxed Hm? does NOT reopen the conversation window (sim: TV cascade Hm? -> window -> next TV fragment answered)
        self.reject(t); return "drop", {**info, "why": "addressee"}

    def barge_verify(self, wav, short=False):
        """barge-in candidate (playback paused): real interruption or TV/talk? -> (ok, info). Stop word / name / question / command pass, other statements need the addressee model (p >= 0.9; short = complete utterance of <1 s, ASR is unreliable there: only p < 0.6 or no ASR text veto)."""
        q = self.asr(wav); info = {"q": q}
        said = " ".join(t for ts, t in getattr(self, "said", []) if time.monotonic() - ts < 60)
        if q.strip() and R.self_echo(q, said): info["echo"] = True; return False, info  # our own words coming back through the speaker ('Ich.')
        if not q.strip(): return short, info
        if R.stop_word(q) or R.barge_stop(q) or R.barge_no(q) or R.barge_contq(q) or R.NAME.search(q) or self.addr is None: return True, info
        qc = R.is_question_or_cmd(q)
        if BARGE_Q == "addr":  # old: question/command passes, else addressee p >= 0.9
            if qc: return True, info
            info["p"] = p = round(self.addr.p_addr(q, self.hist[-2:]), 3); return p >= (0.6 if short else 0.9), info
        sp = " ".join(t for ts, t in getattr(self, "said", [])[-2:])  # what Kitsune is saying right now
        info["pb"] = p = round(self.addr.p_barge(q, sp, self.hist[-1:]), 3); info["qc"] = qc
        return p >= (BARGE_PQ if qc else BARGE_P), info  # sim/barge_eval.py: hits 0.48 -> 0.59, false barge 0.37 -> 0.13 (dev set)

    def reject(self, t):
        self.rejects = [x for x in self.rejects if t - x < 30] + [t]
        if len(self.rejects) >= 3: self.bg_until = t + 60  # background profile: TV/babble around -> be strict for a minute

    # ---- one turn ----
    def turn(self, src, sink=None, now=None, t_end=None, resume=None, barge=False, cont=None):
        """src: wav path / np array (ASR) or ('text', str). t_end: perf_counter of real end of speech (default: turn start).
        resume: watch.Watch (open mic after listen) -> if the user keeps talking before the first audio sample, the turn restarts on the concatenated audio.
        cont: previous transcript when this barge-in capture continues an interrupted sentence. Returns record dict."""
        self.watch = resume; pend0 = (self.pending, self.ignoring, self.mode)
        try: return self._turn(src, sink, now, t_end, barge, cont)
        except Restart:
            extra = resume.collect(); self.watch = None; self.pending, self.ignoring, self.mode = pend0
            t_end = max(t_end or 0.0, resume.t_last or 0.0) or t_end  # real end of the continued speech: the first segment's end made e2e/pre_turn include the whole pause + continuation (log 2026-11-03: e2e 8279 > total 5485)
            return self._turn(np.concatenate([src, extra]), sink, now, t_end, barge, cont, restarted=True)
        finally:
            self.watch = None
            if resume: resume.close()

    def _commit(self):
        """called right before the first audio sample is fed: speculative work is only kept if the user did not resume within the full hangover."""
        w = self.watch
        if w is None: return
        if w.wait_until(self.t_end0 + HANGOVER): raise Restart
        w.close(); self.watch = None

    def _turn(self, src, sink, now, t_end, barge, cont, restarted=False):
        sink = sink or WavSink(); now = now or datetime.now(); self.cancel.clear()  # stale barge-in must not kill this turn
        self.bridge_end(); self.ans_started, self._pe = False, 0.0
        t0 = time.perf_counter(); t_end = t_end or t0; self.t_end0 = t_end; m = {}; rec = {"ts": now.isoformat(timespec="seconds")}
        if restarted: m["resumed"] = True
        if not (self.handsfree and self.mode == "standby"): T.prefetch_kal()  # calendar reads (0.8 s subprocess) overlap ASR+routing
        if isinstance(src, tuple): q = src[1]; m["asr_ms"] = 0
        else:
            q = self.asr(src); m["asr_ms"] = ms(t0, time.perf_counter())
            if (q2 := LX.apply(q)) != q: m["lex"] = q; q = q2  # spelled letters / lexicon variants -> canonical (original ASR text kept in ms.lex)
            if cont: q = cont + " " + q; m["cont"] = True; self.pending = self.pend_prev; (self.hist.pop() if self.hist else None)
            if (w := self.watch) is not None:
                skip = self.handsfree and self.mode == "standby" and not R.NAME.search(q)
                pend = self.pending; hold = 0.0
                if not skip: hold = 1.2 if R.incomplete(q) else (1.0 if pend and R.pending_decide(pend["stage"], q) is None else 0.0)
                if w.speech.is_set() or (hold and w.wait_until(t_end + hold)): raise Restart
        self.pend_prev = self.pending
        rec["transcript"] = q; answer = ""
        t1 = time.perf_counter(); keep = True
        say, brain = None, None  # say: fixed text; brain: (q, tool_out, name)
        pend, self.pending = self.pending, None
        self.d1_res = None; verdict, ginfo = self.gate(q, pend, barge); end_after = False
        if ginfo: rec["gate"] = ginfo
        if verdict == "pass" and ginfo.get("woke"): q = (lambda s: s if len(R.W(s)) >= 4 else q)(R.strip_name(q) or q)  # name + request in one utterance (short rest keeps the name: 'Kitsune, bist du da?' is smalltalk)
        if self.ignoring and R.wakes(q) and not R.not_addressed(q): self.ignoring = False
        if verdict == "drop":
            rec["tool"], say, keep = "standby_drop" if ginfo.get("why") == "standby" else "gate_drop", "", False; m["route_ms"] = 0; self.pending = pend
        elif verdict == "hm": rec["tool"], say, keep = "gate_hm", "Hm?", False; m["route_ms"] = 0; self.pending = pend
        elif verdict == "wake": rec["tool"], say, keep = "wake", "Ja?", False; m["route_ms"] = 0
        elif verdict == "end":
            rec["tool"], say = ("end", "Okay, ich bin still.") if R.name_at(q) else ("antwort", "Gern geschehen."); keep, end_after = False, True; m["route_ms"] = 0
        elif barge and (R.barge_stop(q) or (R.barge_no(q) and len(R.W(q)) <= 2)) and not R.NAME.search(q):  # 'Okay, nein, nein, nein, stopp.' after a barge-in: just stop talking
            rec["tool"], say, keep = "barge_stop", "", False; m["route_ms"] = 0; self.pending = pend
        elif barge and not pend and R.false_start(q):  # fragment after barge-in ('oder', 'Sport.', 'ja'): not a query
            rec["tool"], say, keep = "barge_false", ("" if R.stop_word(q) else "Ja?"), False; m["route_ms"] = 0
        elif self.ignoring:
            rec["tool"], say, keep = "ignored", "", False; m["route_ms"] = 0; self.pending = pend
        elif R.not_addressed(q):
            self.ignoring = True; rec["tool"], say, keep = "ignore", "Okay.", False; m["route_ms"] = 0
        elif pend and (r := self.handle_pending(pend, q, now, lambda: self.filler("eintragen", sink, m, t0))) is not None:
            rec["tool"], say = "pending:" + pend["stage"], r
            m["route_ms"] = 0
        else:
            self.ignoring = False
            ordn = re.search(r"\b(erste|zweite|dritte|vierte|fünfte)[nrs]?\b", q.lower())
            if self.last_items and ordn and re.search(r"meldung|artikel|treffer|nachricht|lies|vor", q.lower()):  # deterministic follow-up
                name, args = "netz_lies", {"url": ordn.group(1)}
            elif pend and pend.get("stage") == "suche_q" and len(R.W(q)) <= 10: name, args = "netz_suche", {"frage": R.strip_cont(q)}
            elif (tc := R.tempo_cmd(q)): name, args = "antwort", {"text": self.set_tempo(tc)}; m["route_ms"] = 0; m["tempo"] = self.tempo
            elif (fu := self.followup_search(q)): name, args = fu; m["followup"] = True
            elif (lw := getattr(self, "last_wx", None)) and time.monotonic() - lw[1] <= 120 and (el := R.ellipsis(q)):  # 'und in Basel?' / 'und morgen?' right after a weather answer
                name, args = "wetter", {**lw[0], **el}; m["followup"] = True
            elif len(R.W(q)) <= 2 and R.NAME.search(q): name, args = "antwort", {"text": "Ja?"}
            elif len(R.W(q)) <= 7 and re.search(r"\b(bist|hörst|verstehst) du( mich)?\b", q.lower()) and not re.search(r"wetter|termin|kalender", q.lower()): name, args = "plaudern", {}  # questions to the assistant itself

            else: name, args = self.route_final(q, now, m)
            m["route_ms"] = ms(t1, time.perf_counter()); t2 = time.perf_counter()
            rec["tool"], rec["args"] = name, args
            if name in ("netz_suche", "wiki_suche"): self.last_search = (name, time.monotonic())
            self.last_wx = ({k_: v for k_, v in args.items() if k_ in ("ort", "wann")}, time.monotonic()) if name == "wetter" else None
            say, brain = self.run_tool(name, args, q, now, sink, m, t0)
            m["tool_ms"] = ms(t2, time.perf_counter())
        if say is not None:
            answer = say; self.speak(say, sink, m, t0)
            if self.foreign: ftxt, flang = self.foreign; self.foreign = None; self.speak(ftxt, sink, m, t0, lang=flang)
        else:
            qq, tout, nm = brain; tb = time.perf_counter(); m["brain_start_ms"] = ms(t0, tb); m["tool_chars"] = len(tout or "")
            if nm in (None, "wiki_suche") and "filler" not in m and not self.cancel.is_set(): self.lead(qq, sink, m, t0)
            buf, spoken = "", False
            out = self._Pipe(self, sink, m, t0) if PIPE else self._Inline(self, sink, m, t0)  # PIPE: TTS of clause n runs in a worker while the brain already writes clause n+1
            try:
                for chunk in (TK.talk(self, qq) if TALKER == "lfm" and nm == "plaudern" else self.brain_stream(self.brain_msgs(qq, tout, nm), 40 if nm == "plaudern" else 160)):
                    if self.cancel.is_set() or out.err: break
                    m.setdefault("brain_first_tok_ms", ms(tb, time.perf_counter()))
                    buf += chunk; answer += chunk
                    if SENT.match(buf) or CLAUSE.match(buf) or len(buf.split()) >= 10: m.setdefault("first_clause_ms", ms(t0, time.perf_counter()))  # first speakable text ready
                    if not spoken and not SENT.match(buf) and not CLAUSE.match(buf) and len(buf.split()) >= 10 and buf.endswith(" "):  # no punctuation yet: flush the first 10 words to cut the lead-in gap
                        out.put(buf.strip()); spoken = True; buf = ""
                    while (mt := (SENT.match(buf) or (CLAUSE.match(buf) if not spoken else None))):
                        out.put(mt.group(1).strip()); spoken = True; buf = buf[mt.end():]
                if buf.strip() and not self.cancel.is_set() and not out.err: out.put(buf.strip())
            finally: out.close()
            m["brain_ms"] = ms(tb, time.perf_counter())
        self.bridge_end(); answer = answer.strip()
        if self.cancel.is_set() and hasattr(sink, "frac_played"):  # history = what the user actually heard
            f = sink.frac_played(); answer = answer[:max(1, int(len(answer) * f))].rstrip() + " … (unterbrochen)"
            rec["barge"] = round(f, 2)
        rec["answer"] = answer
        if getattr(self, "ipa", None): rec["ipa"], self.ipa = self.ipa, None
        if end_after: self.set_mode("standby")
        if keep:
            self.hist.append((q, answer)); self.hist = self.hist[-self.n_hist:]; self.prev_user = q
            if rec.get("tool"): self.topics = (self.topics + [C.topic(rec["tool"], rec.get("args"), q)])[-12:]
        m["total_ms"] = ms(t0, time.perf_counter()); m["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
        if sink.t_first: m["e2e_feed_ms"] = ms(t_end, sink.t_first)  # real end of speech -> first sample handed to the output
        if sink.t_play or sink.t_first: m["e2e_first_audio_ms"] = ms(t_end, sink.t_play or sink.t_first)  # ... -> first audible sample (speaker: incl. pre-buffer + device latency)
        if t_end < t0: m["pre_turn_ms"] = ms(t_end, t0)
        if (su := getattr(self, "setup", None)): m.update({k_: v for k_, v in su.items() if k_ != "t_listen"}); m["listen_to_turn_ms"] = ms(su["t_listen"], t0); self.setup = None  # where did pre_turn_ms go: listen() return -> turn start = sink open + barge start + watch
        rec["ms"] = m; rec["audio_s"] = round(len(sink.audio()) / 24000, 2)
        if hasattr(sink, "under"):
            rec["underruns"], rec["xruns"] = sink.under, sink.xruns  # >0 = audible gaps (stutter)
            self.adapt_pre(bool(sink.under or sink.xruns))
        LOGS.mkdir(exist_ok=True)
        with open(LOGS / f"{now:%Y-%m-%d}.jsonl", "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def gpu_free(self):
        """cached `gpu-run --status` (<= once per 30 s): True when no other job holds the GPU."""
        if time.monotonic() - self.gpu_t > 30:
            self.gpu_t = time.monotonic(); rc, o = T.run(["gpu-run", "--status"], timeout=3); self.gpu_free_ = rc == 0 and "GPU frei" in o
        return self.gpu_free_

    def adapt_pre(self, bad):
        """sink pre-buffer: 0.35 after dropouts, 0.12 after 3 clean turns while the GPU queue is free, else 0.25."""
        if bad: self.clean_turns = 0; PRE[0] = 0.35; return
        self.clean_turns += 1
        PRE[0] = 0.12 if self.clean_turns >= 3 and self.gpu_free() else 0.25

    def interrupted(self, frac):
        """barge-in hit after turn() returned (answer was still playing): history keeps only what was heard."""
        if self.hist and frac < 1:
            q, a = self.hist[-1]; self.hist[-1] = (q, a[:max(1, int(len(a) * frac))].rstrip() + " … (unterbrochen)")

    # ---- tools ----
    def followup_search(self, q):
        """'und zwar KimiC' within 2 min after a search turn: fills in / corrects the topic of that search -> (tool, args) or None."""
        ls = getattr(self, "last_search", None)
        if not ls or time.monotonic() - ls[1] > 120 or not R.is_continuation(q) or not re.match(r"(und )?(zwar|nämlich|ich meine|ich mein|genauer)", q.strip(), re.I): return None
        x = R.strip_cont(q)
        return (ls[0], {"frage": x}) if x and len(R.W(x)) <= 8 else None

    def safe_query(self, arg, utter):
        """web queries leave the machine: only words the user said (this + previous utterance), never assistant/tool text."""
        ok = set(re.findall(r"\w+", (utter + " " + self.prev_user).lower()))
        w = re.findall(r"\w+", arg.lower())
        return arg if w and all(x in ok or len(x) < 3 for x in w) else utter

    def run_tool(self, name, a, utter, now, sink, m, t0):
        """exec_tool in a thread; play the pre-rendered filler if the tool is slow (web: at once, local: after 150 ms)."""
        fk = FILL.get(name)
        if not fk: return self.exec_tool(name, a, utter, now)
        fut = self.pool.submit(self.exec_tool, name, a, utter, now)
        if not concurrent.futures.wait([fut], timeout=fk[1]).done: self.filler(fk[0], sink, m, t0); self.bridge_start(sink, m, t0)
        return fut.result()

    def exec_tool(self, name, a, utter, now):
        """-> (say, brain) exactly one is not None. Templates (say) bypass Gemma."""
        g = lambda k: (a.get(k) or "").strip()
        if name == "antwort": return (g("text") or "Ja?"), None
        if name == "netz_ask": self.pending = {"stage": "suche_q", "ask": "Wonach soll ich suchen?"}; return "Wonach soll ich suchen?", None
        if name == "plaudern": return None, (utter, None, "plaudern")
        if name == "wissen": return None, (g("frage") or utter, None, None)
        if name == "kal_heute": return T.say_kal(T.t_kal("today"), "today", now), None
        if name == "kal_woche": return T.say_kal(T.t_kal("week"), "week", now), None
        if name == "netz_news":
            out, items = T.t_news(g("thema") or None); self.last_items = items; return None, (utter, out, name)
        if name == "netz_suche":
            out, items = T.t_suche(q := self.safe_query(g("frage"), utter)); self.last_items = items
            return None, (utter, out, name)
        if name == "netz_lies": return None, (utter, T.t_lies(g("url"), self.last_items), name)
        if name == "wiki_suche": return None, (g("frage") or utter, T.t_wiki(g("frage") or utter), name)
        if name == "bibliothek": return None, (g("frage") or utter, T.t_buch(g("frage") or utter), name)
        if name == "fakt": return T.say_fakt(T.t_fakt(g("frage") or utter)), None
        if name == "gpu_status": return T.say_gpu(T.t_gpu()), None
        if name == "dict":
            say, self.foreign, self.ipa = T.say_dict(g("wort"), g("nach") or "bg"); return say, None
        if name == "wetter": return self.wetter_step(g("ort"), g("wann"))
        if name == "kal_eintragen":
            res = D.resolve(utter, now)  # the transcript is the ground truth; the router's `wann` only as fallback
            if res.date is None and res.time is None and g("wann"): res = D.resolve(g("wann"), now)
            titel = g("titel"); titel = None if R.title_missing(titel, utter) else titel  # 'Trag morgen um halb drei' -> ask what
            return self.eintragen_step({"titel": titel, "res": res}, now), None
        return None, (utter, None, None)

    def wetter_step(self, ort, wann, tried=False):
        ort = re.sub(r"^(in|im|am|bei|für)\s+", "", ort.strip(), flags=re.I)
        if not ort or re.search(r"\bmein(em|en|er)?\b|\bhier\b", ort.lower()):
            if C.ort(getattr(self, "last_place", "")) and not ort: ort = C.ort(self.last_place)  # default place: KITSUNE_ORT or the last weather place
            else:
                self.pending = {"stage": "ort", "wann": wann, "ask": "Für welchen Ort?"}; return "Für welchen Ort?", None
        try:  # ASR place correction against the fakt place index: >=0.8 single candidate -> silent, 0.6-0.8 -> ask once
            import places
            kind, lab = places.decide(places.cands(ort))
        except Exception as e: kind, lab = "none", None
        if kind == "use" and lab: ort = lab
        elif kind == "ask" and lab and not tried:
            self.pending = {"stage": "ort_confirm", "cand": lab, "wann": wann, "ask": f"Meinst du {lab}?"}; return f"Meinst du {lab}?", None
        wk = bool(re.search(r"montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag|wochenende", wann.lower()))
        js = T.t_wetter(ort, 7 if wk else 3)
        if isinstance(js, str):
            if js.startswith("Den Ort") and not tried:
                self.pending = {"stage": "ort", "wann": wann, "ask": "Für welchen Ort?", "tried": True}; return "Den Ort kenne ich nicht. Für welchen Ort?", None
            return js, None
        self.last_place = js.get("ort") or ort
        return (T.say_wetter(js, wann) or "Dazu habe ich keine Vorhersage."), None

    def eintragen_step(self, p, now):
        res = p["res"]
        if not p.get("titel"):
            ask = "Was soll ich eintragen?"; self.pending = {**p, "stage": "what", "ask": ask}; return ask
        if res.date is None:
            ask = "Für welchen Tag soll ich das eintragen?"; self.pending = {**p, "stage": "when", "ask": ask}; return ask
        if res.time is None and not res.allday:
            ask = "Um wie viel Uhr?"; self.pending = {**p, "stage": "when", "ask": ask}; return ask
        ask = f"Soll ich {p['titel']} {D.speak(res, now)} eintragen?"
        self.pending = {**p, "stage": "confirm", "ask": ask}
        return ask

    def handle_pending(self, p, q, now, filler=lambda: None):
        """-> spoken reply, or None if the utterance is a new request."""
        d = R.pending_decide(p["stage"], q)
        if d == "status" and p["stage"] != "ort":  # 'hast du's schon gemacht?' -> nothing was written yet: ask again
            self.pending = p; return "Noch nicht. " + p["ask"]
        if p["stage"] == "ort_confirm":
            yn = R.yes_no(q)
            if yn: return self.wetter_step(p["cand"], p.get("wann", ""), tried=True)[0]
            if yn is False: self.pending = {"stage": "ort", "wann": p.get("wann", ""), "ask": "Für welchen Ort?", "tried": True}; return "Okay. Für welchen Ort?"
            if len(R.W(q)) > 4: return None
            return self.wetter_step(re.sub(r"[.?!]", "", q).strip(), p.get("wann", ""), tried=True)[0]
        if p["stage"] == "ort":
            if len(R.W(q)) > 4: return None
            return self.wetter_step(re.sub(r"[.?!]", "", q).strip(), p.get("wann", ""), tried=p.get("tried", False))[0]
        if p["stage"] == "confirm":
            if d is None: return None
            if d == "no": return "Okay, dann lasse ich es."
            filler()
            out, ok = T.t_kal_add(p["titel"], D.kal_at(p["res"]))
            self.last_action = out
            return ("Okay, ist eingetragen." if not self.dry else "Okay, eingetragen. Das war ein Probelauf.") if ok else "Das hat nicht geklappt."
        if d == "no": return "Okay, dann lasse ich es."
        if p["stage"] == "suche_q": return None
        if p["stage"] == "what":
            ws = R.W(q)
            if len(ws) > 7 or R.title_missing(q): return None
            return self.eintragen_step({**p, "titel": re.sub(r"\s+(ein|rein|bitte)[.!]?$", "", q.strip(" .!?"), flags=re.I)}, now)
        r2 = D.resolve(q, now)
        if not r2.ok or (r2.date is None and r2.time is None): return None
        old = p["res"]
        res = dataclasses.replace(old, date=old.date or r2.date, time=old.time or r2.time, ok=True)
        return self.eintragen_step({**p, "res": res}, now)
