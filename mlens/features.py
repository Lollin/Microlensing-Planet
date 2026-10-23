"""Features invariantes a translação/escala: ajusta PSPL (1L1S) e usa o RESÍDUO em unidades de tE.
A assinatura planetária vive no resíduo; é isso que a rede precisa enxergar."""
import numpy as np
from scipy.optimize import least_squares
from . import config as C

NB, TAU = 256, 4.0                      # bins e meia-largura da janela em unidades de tE


def _A(t, t0, u0, tE):
    u2 = u0 ** 2 + ((t - t0) / tE) ** 2
    return (u2 + 2) / np.sqrt(u2 * (u2 + 4))


def _lin(A, f, w):                      # fluxo = fb + fs*A  (mínimos quadrados ponderados, forma fechada)
    S = np.stack([A, np.ones_like(A)], 1); W = S * w[:, None]
    th = np.linalg.lstsq(W.T @ S, W.T @ f, rcond=None)[0]
    return S @ th, th


def pspl_fit(t, f, s):
    w = 1.0 / s ** 2
    def res(q):
        m, _ = _lin(_A(t, q[0], np.exp(q[1]), np.exp(q[2])), f, w); return (f - m) / s
    nb = 100; e = np.linspace(t.min(), t.max(), nb + 1); idx = np.clip(np.digitize(t, e) - 1, 0, nb - 1)
    bm = np.bincount(idx, f, nb) / np.maximum(np.bincount(idx, minlength=nb), 1)
    t0g = 0.5 * (e[1:] + e[:-1])[np.argmax(bm)]
    best = None
    for u0 in (0.1, 0.4, 1.0):
        for tE in (10., 25., 50.):
            q = np.array([t0g, np.log(u0), np.log(tE)]); c = np.sum(res(q) ** 2)
            if best is None or c < best[0]: best = (c, q)
    sol = least_squares(res, best[1], method="lm", max_nfev=60)
    q = sol.x; m, th = _lin(_A(t, q[0], np.exp(q[1]), np.exp(q[2])), f, w)
    return dict(t0=q[0], u0=np.exp(q[1]), tE=np.exp(q[2]), fs=th[0], fb=th[1], model=m, chi2=float(np.sum(((f - m) / s) ** 2)))


def featurize(data):
    """data = {sítio: (t, fluxo, erro)}  ->  (X[3,NB], escalares[5], dchi2)"""
    t = np.concatenate([d[0] for d in data.values()]); f = np.concatenate([d[1] for d in data.values()])
    s = np.concatenate([d[2] for d in data.values()])
    fit = pspl_fit(t, f, s); tau = (t - fit["t0"]) / fit["tE"]; r = (f - fit["model"]) / s
    edges = np.linspace(-TAU, TAU, NB + 1); k = np.digitize(tau, edges) - 1; ok = (k >= 0) & (k < NB)
    n = np.bincount(k[ok], minlength=NB); cnt = np.maximum(n, 1)
    fl = np.bincount(k[ok], f[ok], NB) / cnt
    z = np.bincount(k[ok], r[ok], NB) / np.sqrt(cnt)               # z-score do resíduo médio por bin
    mask = (n > 0).astype(float)
    X = np.stack([np.where(mask > 0, fl, 1.0), np.clip(z, -20, 20), mask]).astype(np.float32)
    dchi2 = fit["chi2"] - len(t)                                   # ~ Δχ² vs. modelo verdadeiro (χ²_verd ≈ N)
    sc = np.array([np.log10(max(dchi2, 1.0)), np.log10(fit["tE"]), np.log10(fit["u0"]), fit["fs"], (fit["t0"] - C.T_MIN) / (C.T_MAX - C.T_MIN)], np.float32)
    return X, sc, dchi2, fit
