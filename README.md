<p align="center"><img src="docs/banner.jpg" alt="Kitsune" width="100%"></p>

# Kitsune

A fully local, German-speaking voice assistant for Apple Silicon. LFM3-Audio listens and speaks, a small Gemma model
does the thinking, tools (calendar, weather, news, web search, Wikipedia, facts, dictionary) are called by a router.
Everything runs offline except the explicit network tools. Hands-free by default: it sits in STANDBY until it hears its
name, then holds a conversation, and stops talking when you interrupt it (barge-in).

> Status: personal research project, not a polished product. German-first (prompts, rules and test sets are German).
> Model weights are not included.

## How it works

```
mic -> VAD -> ASR (LFM3-Audio) -> wake/addressee gate (d1-2B) -> router (LFM3 or d1) -> tool or Gemma brain
    -> speakable text (digits, dates -> words) -> TTS (LFM3-Audio, streaming codec) -> speaker
         ^ barge-in detector + echo/self-hearing veto + optional Apple Voice Processing AEC (aec/, Swift)
```

Highlights: standby/active conversation states, addressee gate that ignores TV and third-party talk, speculative
endpointing, KV-prefix-cached router, pre-rendered fillers, pitch-preserving tempo control, ASR place-name correction,
pronunciation/ASR lexicon, and a silent simulation harness (`sim/`) with a virtual room (image-source RIR, noise beds)
that drives the real audio code through a fake `sounddevice`.

## Measured numbers (from the notes below)

- Wake word recall with an ASR name hint: 6/10 to 9/10.
- Tool routing on a held-out set: 28/30 (LFM3 router) and 28/30 (d1 router); facts-not-chat 14/18 vs 16/18; median routing time 275 ms vs 242 ms.
- Barge-in verification (84 hand-written fragments, dev set): hit rate 0.48 to 0.59, false barge-ins 0.37 to 0.13.
- Addressee gate: about 75 ms per decision; threshold eval on 88 labelled lines.

## Setup

Requires an Apple Silicon Mac, Python 3.11+, and the model weights downloaded by you:

| Role | Model (expected under `~/AI-Model/models/` unless noted) |
|---|---|
| ASR + TTS | LiquidAI LFM3-Audio-2B (MLX 8 bit) |
| Addressee gate / router | LiquidAI d1-2B (MLX 8 bit) |
| Brain | Gemma 4 E4B (MLX 4 bit) |
| Test voice for `sim/` | Piper Thorsten (de_DE-thorsten-medium) |

Each model has its own license, which applies to its use and output.

```bash
python3 -m venv ~/.venvs/mlxvlm && ~/.venvs/mlxvlm/bin/pip install -r requirements.txt
export KITSUNE_PY=~/.venvs/mlxvlm/bin/python          # interpreter used by ./kitsune and the scripts (default as shown)
export KITSUNE_MLXVLM_SRC=~/path/to/mlx-vlm-fork      # checkout with mlx_vlm.models.lfm3_audio
export KITSUNE_PROFILE="Nutzer: ..."                  # optional one-line user profile for the brain
aec/build.sh                                          # optional: Swift Apple Voice Processing helper
./kitsune --help
```

Notes: `./kitsune` wraps the process in a `gpu-run` lease (a small local GPU/RAM queue script from my machine). If you do not
have it, replace `gpu-run -n kitsune --ram 15 ... ` in `kitsune` with a direct interpreter call. The tools shell out to optional
companion CLIs if present (`kal` calendar, `wiki`, `fakt`, `dict`, `buch`, `netz`); missing ones are skipped.
Unit tests that need no models: `python -I -m unittest discover -s . -p 'test_rules.py'` (also `test_dates.py`, `test_gate.py`).

## License

MIT, see `LICENSE`.

---

# Development log (raw notes, newest areas first-ish)
**States (2026-11-01 v2):** hands-free starts in STANDBY: every VAD segment is ASR'd but silently dropped (no router, brain or audio) unless the first words contain the name 'Kitsune' (fuzzy phonetic + edit-distance matcher `rules.name_at`: Kizune, Kizude, Kitsume, Isune, Zune, Zunge ...). Name only -> "Ja?" and ACTIVE; name + request -> answered directly. ACTIVE = conversation mode, no name needed; back to STANDBY after `KITSUNE_STANDBY_S` (default 45, 0 = never) without interaction, or on "Kitsune, Pause/Schluss" ("Okay, ich bin still.") or "danke, das war's". The ASR system prompt carries the hint 'The assistant is called Kitsune.' (`KITSUNE_ASR_HINT=0` disables; wake recall 6/10 -> 9/10; echo of the hint on noise is filtered).
In ACTIVE the addressee gate (`addressee.py`, d1-2B 'noul', ~75 ms) drops TV/third-party talk: hard pass = name, pending question, stop word, <=5-word utterance within 8 s (15 s pending) of Kitsune's last audio; hard drop = <3 words without question/command; else p>=`KITSUNE_GATE_ANS` (0.8) answers, p>=`KITSUNE_GATE_HM` (0.6) + question/command word -> "Hm?", else silent. 3 rejects in 30 s -> background profile: p>=`KITSUNE_GATE_BG` (0.9) or name for 60 s. `KITSUNE_GATE=0` turns the gate model off. Threshold eval: `sim/gate_eval.py` (88 labelled lines, sim/data/gate_eval.json).
Endpointing: listen() hands over after `KITSUNE_SPEC_S` (0.25) of silence (speculative ASR+route), `watch.py` keeps polling the open mic every 100 ms; if the user resumes before the first audio sample the turn restarts on the concatenated audio; no audio is fed before the full 0.4 s hangover; hold 1.2 s when the transcript ends on preposition/article/number/'halb' (`rules.incomplete`), 1.0 s while a question is pending. Barge-in capture right after end of speech is merged with the interrupted sentence (continuation).
Other env: `KITSUNE_ORT` default place for weather, `KITSUNE_VOICE`, `KITSUNE_MIC`, `KITSUNE_PRIO`.

Run: `./kitsune` (hands-free: just speak, 0.4 s pause (hangover) ends the turn; ASR runs at once and is redone on the concat if you keep talking, mic off while Kitsune speaks; `--ptt` = Enter push-to-talk; `q`+Enter quits; exits after 10 min idle; default mic = internal MacBook mic, `--mic`/`--mics`). Wraps `gpu-run -n kitsune --ram 13`.
Tests/headless: `./kitsune --dry-run ask --wav wavs/in/u*.wav --json out.json`, `./kitsune text "frage"`, `./kitsune say "text" --out x.wav`. Add `--play` to hear it; `--voice male`; `--brain lfm` for fallback.
Unit tests: `python -I -m unittest discover -s . -p 'test_dates.py'`; codec parity: `gpu-run -n k --ram 8 python -I test_codec.py`.
Logs: logs/YYYY-MM-DD.jsonl. kal_eintragen needs spoken confirmation; `--dry-run` never calls `kal add`.

**GPU priority:** the interactive mic loop (`./kitsune`, no subcommand) runs with `gpu-run --prio` (parallel to other GPU jobs, RAM gate stays). `ask/text/say` queue normally; override with `KITSUNE_PRIO=0|1`. Agents/tests: never set KITSUNE_PRIO=1.


**P1/P2 (2026-11-01):** router KV prefix cache (static system+tools prompt, date moved into the user msg; `route_prefill_ms/route_decode_ms/route_cached` in logs), pre-rendered fillers (`cache/fillers-*.npz`), templates without Gemma for kal/gpu/fakt/wetter (`tools.say_*`), `plaudern` (Gemma persona, <=2 sentences) instead of parroting `antwort`, barge-in fragments -> "Ja?" (`b*.wav` in tests = after barge-in), "nicht an dich gerichtet" -> silent until name/clear question (`rules.py`), status questions during pending confirmation re-ask. Logs: `e2e_first_audio_ms` = real end of speech -> first audible sample, `hangover_ms`. VAD stays RMS (no Silero weights offline; mlx-audio has the model code only). Sink pre-buffer 0.25 s, auto 0.35 s after any underrun/xrun.
Tests: `python -I -m unittest discover -s . -p 'test_rules.py'`; `~/.venvs/mlxvlm/bin/python -I -m unittest discover -s . -p 'test_engine_pending.py'`; full dialogue (dry-run, no audio): `gpu-run -n kitsune --ram 13 ./suite.sh [codedir] [out.json]`; new test wavs: `gpu-run -n k --ram 6 python -I mkwavs.py` (only generates missing).

**sim/ (silent test harness, 2026-11-01):** Piper-Thorsten test voice + virtual room (image-source RIR, noise beds, speaker/mic models) drive the real listen()/BargeIn/SpeakerSink/turn through a fake `sounddevice` (real time, dry-run, no real audio). Run: `gpu-run -n kitsune-sim --ram 12 ~/.venvs/mlxvlm/bin/python -I sim/run.py --rooms buero,kueche,hotel,auto,kopfhoerer --scen all --only-room-core kueche,hotel,auto,kopfhoerer --bias` (`--list`, `--scen core|name,name`, `--voice lfm3`). Scenarios: sim/scenarios.yaml; reports: sim/reports/<ts>.md + json per room; unit test: `python -I -m unittest discover -s sim -p 'test_room.py'`. `sim/measure_room.py <name> --ich-darf-laut` = AUDIBLE sweep with real mic -> sim/rooms/real_<name>.npz (use as `--rooms real_<name>`); not run yet.

**v2 additions:** `ctx.py` one context builder (state line with date/place/pending in the router user message so the router prefix stays static for the KV cache; brain gets self description, profile, running topic summary, last tool-result handle), ASR-awareness text + few-shots in router/brain prompts, `ctx.speakable` (digits -> words), `places.py` ASR place correction before weather (>=0.8 single candidate silent, 0.6-0.8 "Meinst du X?", else "Für welchen Ort?"; max one clarification), kal title missing -> stage `what` ("Was soll ich eintragen?"), tool `dict` (`dict` CLI, German sentence + the foreign word spoken with a TTS language prompt). Sim: scenarios can set `mode: standby|active`, `standby_s`, `gate: true` (no open conversation window at start). Tests: `python -I -m unittest discover -s . -p 'test_*.py'` (engine tests with the mlxvlm venv). Gate eval: `gpu-run -n k-gate --ram 8 ~/.venvs/mlxvlm/bin/python -I sim/gate_eval.py`.

**2026-11-03 live-log fixes:**
- *Self-hearing (variant A):* `barge.py` calibrates G only from blocks >=0.3 s after playback start (12 blocks, median, floor `KITSUNE_G_MIN_SPK` 0.06 on built-in speakers / `KITSUNE_G_MIN` 0.0003 else; sim echo is ~100x weaker than the real MacBook so the sim only sees the tiny floor) and persists G per `mic|output device|output volume` in `cache/room_g.json` (volume change -> new key -> prior or recalibration). `engine.barge_verify` vetoes ASR text that is (a fragment of) Kitsune's own recent speech (`rules.self_echo`; real "Stopp" still passes), logged as `verify.echo`.
- *Apple Voice Processing (opt-in, default OFF):* `aec/main.swift` (build `aec/build.sh`, self-check `aec/kitsune-aec --self-check`, no-audio protocol mode `--null`) + `aec.py`. `./kitsune --aec` or `KITSUNE_AEC=1`: playback and mic go through ONE VPIO unit (AGC off unless `KITSUNE_AEC_AGC=1`, ducking min). Fallback to plain sounddevice if the binary is missing / no mic permission / helper dies. Tests: `python -I -m unittest discover -s . -p test_aec.py` (needs the mlxvlm venv). VAD floor `KITSUNE_VAD_MIN` (0.004) may need 0.003 with AEC.
- *Conversation:* `lexicon.py` + `lexicon.tsv` (variant<TAB>canonical, hot-reloaded; spelled letters 'K I M I C' -> KIMIC first), applied to the ASR text (original in `ms.lex`). Web phrases ('im Web/Internet/Netz', online, google) -> `netz_suche` via `rules.fast_route` (empty topic -> 'Wonach soll ich suchen?'); 'und zwar X' within 2 min after a search re-runs it with X. Gate: continuation openers (und zwar/nämlich/ich meine ...) pass within 25 s after the last real answer; a middling addressee score (0.6-0.8) right after an answer gives 'Hm?' (once per 20 s, never in the TV/background profile).
- *Latency log:* `pre_turn_ms`/`e2e_first_audio_ms` were inflated on resumed turns (user continued after a pause: t_end was the first segment's end). Fixed (t_end = last loud block of the continuation). New fields `sink_open_ms/barge_start_ms/watch_ms/listen_to_turn_ms`. ASR time scales ~0.18 s per s of audio (+ doubled work on restarts).
- Sim: `sim/scenarios_selfhear.yaml` (`calib_scale` simulates a wrong first calibration).

**2026-11-03 router/filler/context (flags default ON unless noted):**
- *Chat default:* `rules.chat_demote` after the router (not after `fast_route`): statements with a chatty start ('Ich habe heute Banitsa gebacken', 'Du bist ein dummer Assistent'), self-directed questions ('Hast du eine Lieblingsfarbe?') and kal_* picks without any calendar cue go to `plaudern` (statements) or `wissen` (question-shaped, so facts never become chat). Off: `KITSUNE_CHAT_DEMOTE=0`. Prompt variant `KITSUNE_ROUTER_CHAT=1` (OFF by default: cost a tool regression). Eval: `HELDOUT=1 gpu-run -n k-router --ram 12 python -I sim/router_eval.py <tag>` (held-out `sim/data/router_heldout.json` + talk_eval + scenario tools; held-out only with HELDOUT=1, never tune on it).
- *Filler chain:* after a filler, a bridge thread chains up to `KITSUNE_FILL_CHAIN` (default 2, 0 = off) short pre-rendered fillers ('Moment noch.', 'Gleich.') only while the sink playhead is about to run dry and no answer audio was fed yet (`Kitsune.bridge_start`, playhead estimate `_pe`). `ms.bridges` in logs. New filler texts change the cache hash: first start re-renders `cache/fillers-*.npz`.
- *Tool context:* `tools.trim_ctx` cuts tool output for Gemma to title+lead of the top hit, further hits shorter, `KITSUNE_CTX_CHARS` (default 800, 0 = old 2500). Eval: `sim/ctx_eval.py`, `sim/fill_gap.py`, `sim/scenarios_fill.yaml`, `sim/fill_report.py`. `KITSUNE_SUCHE_FIXTURE=<json>` fakes `netz suche` in sims (DuckDuckGo blocks bots).

**2026-11-06 d1 router / follow-ups / speakable:**
- *Router:* `KITSUNE_ROUTER=lfm|d1|shadow` (default **shadow**: LFM3 decides, d1's pick logged as `ms.d1_tool/d1_conf/d1_ms`). `d1`: d1-2B `choice` over the tool list (same forward pass as the addressee gate when the gate runs, else ~220 ms); args by rule (no-arg tools, `frage` = transcript, weather via `rules.wx_slots`); LFM3 router for kal_eintragen/dict, conf < `KITSUNE_D1_CONF` (0.6), pronoun follow-ups. A/B (`sim/reports/router/ab-{lfm,d1}.json`): held-out tools 28/30 both, facts-not-chat lfm 14/18 vs d1 16/18; route med 275 vs 242 ms, p90 363 vs 596 (fallback pays both).
- *Weather ellipsis:* 'und in Basel?' / 'und morgen?' / 'Bitte in Berlin.' within 120 s after a weather answer reuse its slots (`rules.ellipsis`, `self.last_wx`); sim `wetter_ellipse`.
- *speakable:* dates/ordinals by case ('am zweiten August', 'der dritte Oktober', 'bis zum ...'), years 1100-1999, millions ('rund drei Komma sieben Millionen'), abbreviations (z. B., ca., Nr., Art., km/h, €, St.); `SENT` no longer splits after '2.' / 'z.' / 'ca.'.
- *Shadow without latency:* in `shadow` the d1 pick is computed after the answer finished playing (one call in the 0.3 s echo pause, queue max 20) and logged as an extra line `{"shadow": {q, lfm, d1_tool, d1_conf, d1_ms, agree}}`; when the gate already ran d1 the pick is free and stays in `ms`. Parallel d1+LFM3 threads give no gain (GPU-bound, `sim/bench_par.py`: serial 571 vs parallel 547 ms).
- *Barge verify (d1 question):* `barge_verify` asks d1 "will der Nutzer unterbrechen/korrigieren?" with Kitsune's current sentence (`addressee.p_barge`); after echo veto + stop rules: p >= `KITSUNE_BARGE_P` (0.45), question/command >= `KITSUNE_BARGE_PQ` (0.25). `KITSUNE_BARGE_Q=addr` = old (question/command always passes, addressee p >= 0.9). Eval `sim/barge_eval.py` (84 hand-written fragments, dev set): hits 0.48 -> 0.59, false barge 0.37 -> 0.13; sim barge/TV 17/18, no false barge.
- *Speech rate:* `KITSUNE_SPEED=fast|normal|slow` (default fast: TTS prompt '..., speaking quickly at a fast pace.', ~+14 %) and `KITSUNE_TEMPO` (default 1.15, pitch-preserving streaming WSOLA in `tempo.py`, ~2 ms CPU per s of audio, fillers stretched once per tempo). Spoken 'sprich schneller/langsamer/normal', 'nicht so schnell' -> `rules.tempo_cmd` -> +-0.1 (0.8-1.5), persisted in `cache/tempo.json`. New voice prompt changes the filler cache hash (first start re-renders).
