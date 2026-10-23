"""Simulação de curvas de luz 2L1S com VBMicrolensing + efeitos observacionais."""
import numpy as np
import VBMicrolensing
from . import config as C

_VBM = VBMicrolensing.VBMicrolensing()
_VBM.Tol = 1e-3


def sample_params(rng, caustic_frac=None):
    """Priori uniforme; com prob. caustic_frac a trajetória é ancorada num ponto aleatório da cáustica (+ jitter).
    Convenção VBM: fonte(τ) = u0·(sinα, −cosα) + τ·(−cosα, −sinα), com τ=(t−t0)/tE."""
    frac = C.CAUSTIC_FRAC if caustic_frac is None else caustic_frac
    p = {k: rng.uniform(*v) for k, v in C.PRIORS.items()}
    if rng.random() >= frac:
        return p
    cs = _VBM.Caustics(10 ** p["logs"], 10 ** p["logq"])
    for _ in range(10):
        k = int(rng.integers(len(cs))); xs, ys = np.asarray(cs[k][0]), np.asarray(cs[k][1])
        i = int(rng.integers(len(xs))); size = max(np.ptp(xs), np.ptp(ys))
        a = p["alpha"]; u0 = xs[i] * np.sin(a) - ys[i] * np.cos(a) + rng.uniform(-0.5, 0.5) * size
        if u0 < 0:
            u0, a = -u0, (a + np.pi) % (2 * np.pi)       # mesma reta, percorrida no sentido oposto
        if 0.005 < u0 < C.U0_MAX_CAUSTIC:
            p["u0"], p["alpha"] = float(u0), float(a); break
    return p


def magnification(p, t):
    # VBM: [log s, log q, u0, alpha, log rho, log tE, t0]  (logs naturais)
    pr = [np.log(10 ** p["logs"]), np.log(10 ** p["logq"]), p["u0"], p["alpha"],
          np.log(C.RHO), np.log(p["tE"]), p["t0"]]
    return np.asarray(_VBM.BinaryLightCurve(pr, t)[0])


def observation_times(rng):
    """Instantes de observação por sítio: janela noturna diária, noites perdidas, lacuna sazonal."""
    gap_len = rng.uniform(*C.SEASON_GAP_DAYS)
    gap_start = rng.uniform(C.T_MIN, C.T_MAX - gap_len)
    out = {}
    for name, s in C.SITES.items():
        ts = []
        for d in range(int(C.T_MIN), int(C.T_MAX)):
            if rng.random() < C.P_BAD_WEATHER:
                continue
            start = d + s["phase_h"] / 24.0
            n = int(C.NIGHT_HOURS / s["cadence_h"])
            ts.append(start + np.arange(n) * s["cadence_h"] / 24.0)
        t = np.concatenate(ts)
        t = t[(t >= C.T_MIN) & (t < C.T_MAX)]
        t = t[(t < gap_start) | (t > gap_start + gap_len)]
        out[name] = t
    return out


def simulate_event(rng, p=None):
    """Retorna dict com params, dados por sítio (t, flux, err) já com ruído gaussiano."""
    p = p or sample_params(rng)
    data = {}
    for name, t in observation_times(rng).items():
        A = magnification(p, t)
        sig = C.SITES[name]["sigma"] * np.sqrt(A)       # erro relativo cresce com sqrt(A)
        flux = A + rng.normal(0.0, sig)
        data[name] = (t, flux, sig)
    return p, data


def to_grid(data):
    """Funde sítios numa grade fixa (N_BINS): canal 0 = fluxo médio (ponderado), canal 1 = máscara."""
    edges = np.linspace(C.T_MIN, C.T_MAX, C.N_BINS + 1)
    num = np.zeros(C.N_BINS); den = np.zeros(C.N_BINS)
    for t, f, s in data.values():
        idx = np.clip(np.digitize(t, edges) - 1, 0, C.N_BINS - 1)
        w = 1.0 / s ** 2
        np.add.at(num, idx, w * f); np.add.at(den, idx, w)
    mask = den > 0
    flux = np.where(mask, num / np.where(mask, den, 1), 1.0)   # lacuna -> linha de base
    return np.stack([flux, mask.astype(float)])


def params_to_vec(p):
    return np.array([p["t0"], p["u0"], p["tE"], p["logq"], p["logs"],
                     np.cos(p["alpha"]), np.sin(p["alpha"])])
