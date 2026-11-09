"""Streaming decode for the Liquid codec (MLX). Port of LiquidCodecModel.decode(codes, streaming_cache, finalize).

State carried between chunks: decoder cache (conv tails of the ShortConv layers, KV window 30 of the SWA layers)
and the ISTFT overlap-add tail (2*trim = 960 samples + window envelope). Chunks concatenate to the full decode.
"""
import mlx.core as mx
from mlx_vlm.models.cache import ArraysCache, RotatingKVCache


class StreamingCodec:
    def __init__(self, codec):
        self.c = codec
        cfg = codec.config
        self.trim = (cfg.stft_win_length - cfg.stft_hop_length) // 2
        self.tail = 2 * self.trim
        self.reset()

    def reset(self):
        lfm = self.c.decoder.lfm
        self.cache = [ArraysCache(size=1) if l.layer_type == "conv" else RotatingKVCache(max_size=lfm.config.sliding_window)
                      for l in lfm.layers]
        self.aud = self.env = None
        self.started = self.finished = False

    def _raw(self, codes):
        """codes [12, n] -> un-normalised overlap-added audio [(n*6-1)*hop+win] and its window envelope."""
        c, d = self.c, self.c.decoder
        x = c.from_quantizer(c.quantizer(codes[None].astype(mx.int32)))
        h = d.lfm(input_embeddings=mx.repeat(x, d.stack, axis=1), cache=self.cache)
        log_mag, ang = mx.split(d.lin(h).astype(mx.float32), 2, axis=-1)
        mag = mx.exp(log_mag)
        spec = mag * mx.cos(ang) + 1j * (mag * mx.sin(ang))
        ist = d.istft
        win = ist.window.astype(mx.float32)
        frames = mx.fft.irfft(spec, n=ist.n_fft, axis=-1) * win
        aud = ist._overlap_add(frames)[0]
        env = ist._overlap_add(mx.broadcast_to(win * win, (1, spec.shape[1], win.shape[0])))[0]
        return aud, env

    def decode(self, codes, finalize=False):
        """codes [12, n] (n may be 0 only with finalize) -> float32 samples [m]; 1920/frame in the long run."""
        assert not self.finished, "stream finalized; call reset()"
        n = 0 if codes is None else codes.shape[-1]
        if n == 0:
            if not (finalize and self.started):
                self.finished |= finalize
                return mx.zeros((0,), mx.float32)
            aud, env = mx.zeros((self.tail,)), mx.zeros((self.tail,))
        else:
            aud, env = self._raw(codes)
        t, tr = self.tail, self.trim
        if self.aud is not None:
            aud = mx.concatenate([aud[:t] + self.aud, aud[t:]])
            env = mx.concatenate([env[:t] + self.env, env[t:]])
        if not self.started:
            aud, env = aud[tr:], env[tr:]
            self.started = True
        if finalize:
            aud, env = aud[:-tr], env[:-tr]
            self.aud = self.env = None
            self.finished = True
        else:
            self.aud, self.env = aud[-t:], env[-t:]
            aud, env = aud[:-t], env[:-t]
        out = aud / env
        return out
