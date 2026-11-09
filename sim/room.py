"""Virtual acoustic room: shoebox image-source RIR (Eyring absorption) + diffuse tail, device models, presets. numpy/scipy only."""
import numpy as np
from dataclasses import dataclass, field
from scipy import signal

SR, C = 16000, 343.0
USER_RMS = 0.017          # active-speech RMS of the user at the mic (block peaks ~0.025 as in the real logs)
MIC_NOISE = 0.0003        # measured mic floor (RMS)


def rir(dim, src, mic, rt60, fs=SR, order=10, t_cut=0.05, seed=0):
    """Image sources (Allen-Berkley) up to t_cut, then exponentially decaying noise tail with the diffuse-field energy. Direct amplitude = 1/d."""
    L, src, mic = np.asarray(dim, float), np.asarray(src, float), np.asarray(mic, float)
    V, S = L.prod(), 2 * (L[0] * L[1] + L[0] * L[2] + L[1] * L[2])
    alpha = 1 - np.exp(-0.161 * V / (S * rt60))  # Eyring
    beta = np.sqrt(1 - alpha)
    n = int(fs * (rt60 + 0.05)); h = np.zeros(n + 16)
    nmax = np.ceil(C * t_cut / (2 * L)).astype(int) + 1
    g = [np.arange(-m, m + 1) for m in nmax]
    nx, ny, nz = np.meshgrid(*g, indexing="ij")
    ts, ds, rs = [], [], []
    for qx in (0, 1):
        for qy in (0, 1):
            for qz in (0, 1):
                q = np.array([qx, qy, qz])
                pos = [(1 - 2 * q[i]) * src[i] + 2 * nn * L[i] for i, nn in enumerate((nx, ny, nz))]
                d = np.sqrt(sum((p - mic[i]) ** 2 for i, p in enumerate(pos)))
                refl = sum(np.abs(nn - q[i]) + np.abs(nn) for i, nn in enumerate((nx, ny, nz)))
                m = (refl <= order) & (d / C <= t_cut)
                ds.append(d[m]); rs.append(refl[m])
    d, refl = np.concatenate(ds), np.concatenate(rs)
    amp = beta ** refl / np.maximum(d, 0.05)
    t = d / C * fs; i0 = np.floor(t).astype(int); fr = t - i0
    for k in range(-3, 5):  # 8-tap windowed-sinc fractional delay
        w = np.sinc(k - fr) * (0.5 + 0.5 * np.cos(np.pi * (k - fr) / 4.5))
        np.add.at(h, i0 + k, amp * w)
    dd = np.linalg.norm(src - mic); dc = 0.057 * np.sqrt(V / rt60)
    a = 6.908 / rt60; e_rev = 1.0 / dc ** 2
    a0 = np.sqrt(e_rev * 2 * a / fs)
    tt = np.arange(n) / fs; i_cut = int(t_cut * fs)
    tail = np.random.default_rng(seed).standard_normal(n) * a0 * np.exp(-a * tt)
    tail[:i_cut] = 0; ramp = int(0.01 * fs); tail[i_cut:i_cut + ramp] *= np.linspace(0, 1, ramp)
    h[:n] += tail
    return h[:n]


def rt60_est(h, fs=SR):
    """Schroeder backward integration, T20 extrapolated to 60 dB."""
    e = np.cumsum(h[::-1] ** 2)[::-1]; db = 10 * np.log10(e / e[0] + 1e-30)
    i5, i25 = np.argmax(db < -5), np.argmax(db < -25)
    return 3 * (i25 - i5) / fs


def norm(h): return h / np.sqrt((h ** 2).sum())


_HP = signal.butter(2, 80, "hp", fs=SR, output="sos")


def mic_hp(x): return signal.sosfilt(_HP, x)


@dataclass
class Preset:
    name: str
    dim: tuple
    rt60: float
    echo: float = 0.02           # speaker->mic amplitude gain (power 4e-4: matches real barge G_mean 4-6e-4)
    d_user: float = 0.9
    d_spk: float = 0.2
    d_noise: float = 2.5
    note: str = ""


PRESETS = {p.name: p for p in [
    Preset("buero", (4, 3.5, 2.6), 0.35),
    Preset("kueche", (4.5, 3.0, 2.5), 0.70, echo=0.024),
    Preset("hotel", (5, 3.8, 2.5), 0.45),
    Preset("auto", (2.0, 1.5, 1.2), 0.12, d_user=0.7, echo=0.03),
    Preset("kopfhoerer", (4, 3.5, 2.6), 0.35, echo=0.0, note="no speaker->mic path"),
]}


class Room:
    """Impulse responses for user->mic, speaker->mic, noise-source->mic (mic highpass baked in) + speaker model."""
    def __init__(self, p, seed=1):
        if isinstance(p, str):
            if p.startswith("real_") or p.endswith(".npz"): p = self.load_real(p)
            else: p = PRESETS[p]
        self.p = p; r = np.random.default_rng(seed); L = np.array(p.dim)
        mic = L * np.array([0.3, 0.5, 0.0]) + [0, 0, 0.95]   # on a desk
        az = r.uniform(-0.5, 0.5); el = 0.25
        u = mic + p.d_user * np.array([np.cos(az), np.sin(az), el])
        s = mic + p.d_spk * np.array([0.15, -0.97, -0.1])    # laptop speaker next to the mic
        nz = mic + p.d_noise * np.array([0.8, 0.5, 0.15])
        cl = lambda x: np.clip(x, 0.1, L - 0.1)
        if getattr(p, "h_user", None) is not None:           # measured preset
            self.h_user, self.h_spk, self.h_noise = p.h_user, p.h_spk, p.h_user
        else:
            self.h_user = norm(mic_hp(rir(L, cl(u), mic, p.rt60, seed=seed)))
            self.h_spk = norm(mic_hp(rir(L, cl(s), mic, p.rt60, seed=seed + 1)))
            self.h_noise = norm(mic_hp(rir(L, cl(nz), mic, p.rt60, seed=seed + 2)))
        self.echo = p.echo
        self.spk_sos = None if getattr(p, "h_user", None) is not None else signal.butter(2, [200, 10000], "bp", fs=24000, output="sos")  # laptop speaker 200 Hz..10 kHz (measured presets already contain the device)

    @staticmethod
    def load_real(name):
        import pathlib
        f = pathlib.Path(name) if name.endswith(".npz") else pathlib.Path(__file__).parent / "rooms" / f"{name}.npz"
        z = np.load(f)
        p = Preset(f.stem, (4, 3.5, 2.6), float(z["rt60"]) if "rt60" in z else 0.4, echo=float(z["echo"]) if "echo" in z else 0.03)
        p.h_spk = norm(z["h_spk"].astype(float)); p.h_user = norm(z["h_user"].astype(float)) if "h_user" in z else p.h_spk
        return p

    @staticmethod
    def soft_clip(x, drive=2.0): return np.tanh(drive * x) / drive


class SpeakerPath:
    """Stateful chunk processor: 24 kHz sink output -> speaker model -> 16 kHz -> RIR -> echo contribution."""
    def __init__(self, room):
        self.room = room; self.zi = np.zeros((room.spk_sos.shape[0] if room.spk_sos is not None else 1, 2))
        self.h48 = signal.firwin(97, 7200, fs=48000); self.zr = np.zeros(len(self.h48) - 1); self.n48 = 0

    def process(self, x24):
        y = np.asarray(x24, float)
        if self.room.spk_sos is not None: y, self.zi = signal.sosfilt(self.room.spk_sos, Room.soft_clip(y), zi=self.zi)
        up = np.zeros(2 * len(y)); up[::2] = 2 * y
        f, self.zr = signal.lfilter(self.h48, 1, up, zi=self.zr)
        idx = np.arange(self.n48, self.n48 + len(f)); self.n48 += len(f)
        x16 = f[idx % 3 == 0]
        return x16 * self.room.echo
