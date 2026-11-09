"""Spike: German speech-to-speech (INTERLEAVED mode) with LFM3-Audio, offline to WAV files only (nothing is played). python -I sim/lfm_interleaved.py"""
import sys, os, json, time, pathlib, threading
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent)); os.environ.setdefault("KITSUNE_GATE", "0")
import numpy as np, mlx.core as mx
HOME = pathlib.Path.home(); sys.path.insert(0, str(__import__('os').environ.get('KITSUNE_MLXVLM_SRC') or HOME / 'Claude-Workspace/lfm3/mlx-vlm'))
mx.set_cache_limit(2 * 2**30); mx.set_memory_limit(16 * 2**30)
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs, should_add_special_tokens
from mlx_audio.audio_io import write as wav_write
from codec_stream import StreamingCodec
import engine as E
OUT = HOME / "Claude-Workspace/lfm3/out/interleaved-de"; OUT.mkdir(parents=True, exist_ok=True)
A, Ap = load(str(E.LIQ / "LFM3-Audio-2B-MLX-8bit")); sc = StreamingCodec(A.codec()); cfg = A.config
tk = getattr(Ap, "tokenizer", Ap)
WAV = pathlib.Path(__file__).parent.parent / "wavs/in"
ins = [("u17", "Hallo nochmal, Kizune. Wie geht es dir heute?"), ("u18", "Na, das ist ja wunderbar. Wer bist du?"), ("u29", "Wie geht es dir sonst so?"), ("u15", "Und warum ist er bunt?"), ("u28", "Kizune, bist du noch da?")]
SYS = {"en": "Respond with interleaved text and audio.", "de": "Respond with interleaved text and audio in German."}

def run(wav, sysm, max_slots=240, mode="interleaved"):
    p = apply_chat_template(Ap, cfg, [{"role": "system", "content": sysm}, {"role": "user", "content": ""}], num_audios=1)
    inp = prepare_inputs(Ap, images=None, audio=[str(wav)], videos=None, prompts=p, add_special_tokens=should_add_special_tokens(cfg.model_type, Ap))
    ids = inp.pop("input_ids"); inp.pop("attention_mask", None)
    t0 = time.perf_counter(); tf = tt = None; toks, buf = [], []
    sc.reset(); pcm = []
    for tok, codes in A.stream_slots(ids, generation_mode=mode, max_tokens=max_slots, **A.audio_sampling, **inp):
        if codes is None:
            if tt is None: tt = time.perf_counter() - t0
            toks.append(tok); continue
        if codes[0].item() == cfg.audio_eos_code: break
        if tf is None: tf = time.perf_counter() - t0
        buf.append(codes)
        if len(buf) >= 2: pcm.append(np.asarray(sc.decode(mx.stack(buf, axis=-1)))); buf = []
    pcm.append(np.asarray(sc.decode(mx.stack(buf, axis=-1) if buf else None, finalize=True)))
    a = np.concatenate([x for x in pcm if len(x)]) if any(len(x) for x in pcm) else np.zeros(0, np.float32)
    return tk.decode(toks), a, tf, tt, time.perf_counter() - t0

res = []
run(WAV / "u17.wav", SYS["en"], 30)  # warm
for variant in ("en", "de"):
    for name, ref in ins:
        txt, a, tf, tt, tot = run(WAV / f"{name}.wav", SYS[variant])
        r = {"variant": variant, "in": name, "user": ref, "text": txt, "audio_s": round(len(a) / 24000, 2), "first_text_ms": round(tt * 1000) if tt else None, "first_audio_ms": round(tf * 1000) if tf else None, "total_ms": round(tot * 1000)}
        res.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
        if variant == "de" and len(a): wav_write(str(OUT / f"{name}_{variant}.wav"), a.astype(np.float32), samplerate=24000, format="wav"); (OUT / f"{name}_{variant}.txt").write_text(f"user: {ref}\nmodel: {txt}\n")
# ASR pass alone vs concurrent with interleaved generation
asr_t = []
for name, _ in ins:
    t = time.perf_counter(); q = E.Kitsune.asr.__get__(type("K", (), {"A": A, "Ap": Ap})())(str(WAV / f"{name}.wav")); asr_t.append((round((time.perf_counter() - t) * 1000), q))
print("asr alone", asr_t)
cc = []
for name, _ in ins[:3]:
    box = {}
    def asr_thread():
        t = time.perf_counter(); E.Kitsune.asr.__get__(type("K", (), {"A": A, "Ap": Ap})())(str(WAV / f"{name}.wav")); box["asr"] = round((time.perf_counter() - t) * 1000)
    th = threading.Thread(target=asr_thread); th.start(); txt, a, tf, tt, tot = run(WAV / f"{name}.wav", SYS["en"], 60); th.join()
    cc.append({"in": name, "asr_ms_concurrent": box.get("asr"), "first_audio_ms_concurrent": round(tf * 1000) if tf else None})
print("concurrent", cc)
(OUT / "report.json").write_text(json.dumps({"runs": res, "asr_alone": asr_t, "concurrent": cc}, ensure_ascii=False, indent=1))
print("peak GB", round(mx.get_peak_memory() / 1e9, 2))
