"""chunked == full decode parity. run: gpu-run -n kitsune-codec --ram 8 python -I test_codec.py"""
import sys, pathlib, time
sys.path.insert(0, str(__import__('os').environ.get('KITSUNE_MLXVLM_SRC') or str(pathlib.Path.home() / 'Claude-Workspace/lfm3/mlx-vlm'))); sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs, should_add_special_tokens
from mlx_vlm.models.lfm3_audio.codec import load_codec
from codec_stream import StreamingCodec
A, Ap = load(str(pathlib.Path.home() / "AI-Model/models/LiquidAI/LFM3-Audio-2B-MLX-8bit"))
p = apply_chat_template(Ap, A.config, [{"role": "system", "content": "Perform TTS in German using a female voice."}, {"role": "user", "content": "Heute hast du zwei Termine, um zehn Uhr den Zahnarzt und am Abend ein Treffen mit Anna."}])
inp = prepare_inputs(Ap, images=None, audio=None, videos=None, prompts=p, add_special_tokens=should_add_special_tokens(A.config.model_type, Ap))
ids = inp.pop("input_ids"); inp.pop("attention_mask", None)
fr = [c for t, c in A.stream_slots(ids, **A.audio_sampling, **inp) if c is not None and c[0].item() != A.config.audio_eos_code]
codes = mx.stack(fr, axis=-1); print("frames", codes.shape[1])
for name, dt in [("bf16", mx.bfloat16), ("fp32", mx.float32)]:
    c = A.codec() if dt == mx.bfloat16 else load_codec(A.model_path, dtype=mx.float32)
    full = np.asarray(c.decode(codes))
    for ch in (1, 2, 3, 7, 13):
        sc = StreamingCodec(c); out = []; n = codes.shape[1]
        for i in range(0, n, ch): out.append(np.asarray(sc.decode(codes[:, i:i + ch], finalize=False)))
        out.append(np.asarray(sc.decode(None, finalize=True)))
        o = np.concatenate(out)
        print(f"{name} chunk={ch}: len {len(o)} vs {len(full)}  max|diff| {np.abs(o - full).max():.2e}  (full max {np.abs(full).max():.2f})")
    # per-frame decode cost
    sc = StreamingCodec(c); t = time.perf_counter()
    for i in range(min(n, 20)): mx.eval(sc.decode(codes[:, i:i + 1]))
    print(name, "ms/frame (chunk=1):", round((time.perf_counter() - t) * 1000 / min(n, 20), 1))
