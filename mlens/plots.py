"""Figuras do banner. Estilo grande/legível para impressão (fontes >= 14 pt)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C
from .model import NAMES

plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18,
                     "legend.fontsize": 14, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"Chile": "#1f77b4", "Africa do Sul": "#d62728", "Australia": "#2ca02c"}


def fig_example(p, data, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, (t, f, s) in data.items():
        ax.errorbar(t, f, s, fmt=".", ms=3, lw=0.4, alpha=0.6, color=COL[name], label=name)
    ax.set(xlabel="Tempo (dias)", ylabel="Amplificação (fluxo normalizado)")
    ax.set_title(f"2L1S sintético: q=10$^{{{p['logq']:.1f}}}$, s={10**p['logs']:.2f}, "
                 f"u$_0$={p['u0']:.2f}, t$_E$={p['tE']:.0f} d")
    ax.legend(markerscale=3); fig.savefig(path); plt.close(fig)


def fig_pred_true(y, yp, r2, path):
    labs = ["t$_0$ (d)", "u$_0$", "t$_E$ (d)", "log q", "log s"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.2)); axs = axs.ravel()
    for ax, i, lab in zip(axs, range(5), labs):
        ax.scatter(y[:, i], yp[:, i], s=5, alpha=0.35, color="#1f4e79", rasterized=True)
        lo, hi = y[:, i].min(), y[:, i].max(); ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set(xlabel=f"{lab} verdadeiro", ylabel="previsto"); ax.set_title(f"R² = {r2[NAMES[i]]:.2f}")
    a, ap = np.arctan2(y[:, 6], y[:, 5]) % (2*np.pi), np.arctan2(yp[:, 6], yp[:, 5]) % (2*np.pi)
    ax = axs[5]; ax.scatter(a, ap, s=5, alpha=0.35, color="#8b1a1a", rasterized=True)
    ax.plot([0, 2*np.pi], [0, 2*np.pi], "k--", lw=1.2)
    ax.set(xlabel="α verdadeiro (rad)", ylabel="previsto"); ax.set_title(f"erro médio = {r2['alpha_erro_medio_deg']:.0f}°")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_geometry(path, s=1.1, q=1e-3, u0=0.12, alpha=0.9, tE=30.0, t0=0.0):
    """Cáusticas da lente binária + trajetória da fonte + curva de luz correspondente."""
    import VBMicrolensing
    V = VBMicrolensing.VBMicrolensing(); V.Tol = 1e-3
    t = np.linspace(-1.6 * tE, 1.6 * tE, 1500)
    pr = [np.log(s), np.log(q), u0, alpha, np.log(C.RHO), np.log(tE), t0]
    A, y1, y2 = V.BinaryLightCurve(pr, t)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for c in V.Caustics(s, q):
        a1.plot(c[0], c[1], color="#d62728", lw=2)
    a1.plot(y1, y2, color="#1f4e79", lw=1.8); a1.annotate("", xy=(y1[-1], y2[-1]), xytext=(y1[-40], y2[-40]),
        arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=2))
    a1.plot([-s*q/(1+q), s/(1+q)], [0, 0], "ko", ms=[9, 4][0]); a1.set_aspect("equal")
    a1.set(xlim=(-0.8, 1.6), ylim=(-0.9, 0.9), xlabel="x (θ$_E$)", ylabel="y (θ$_E$)")
    a1.set_title(f"Cáustica e trajetória (s={s}, q=10$^{{{np.log10(q):.0f}}}$, α={np.degrees(alpha):.0f}°)", fontsize=15)
    a2.plot(t, A, color="k", lw=1.8); a2.set(xlabel="Tempo (dias)", ylabel="Amplificação")
    a2.set_title("Curva de luz resultante (sem ruído)", fontsize=15); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_r2_bars(r2_1l1s, r2_2l1s, path):
    """r2_1l1s: dict dos valores do resumo; r2_2l1s: dict desta etapa (chaves comuns: t0,u0,tE)."""
    keys = list(r2_2l1s); x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, [r2_1l1s.get(k, np.nan) for k in keys], w, label="1L1S (CNN)", color="#7f7f7f")
    ax.bar(x + w / 2, [r2_2l1s[k] for k in keys], w, label="2L1S (RNA)", color="#1f4e79")
    ax.set_xticks(x, keys); ax.set(ylabel="R²", ylim=(0, 1.05)); ax.legend(); fig.savefig(path); plt.close(fig)


def fig_loss(model, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(model["net"].loss_curve_, label="treino")
    ax.set(xlabel="Época", ylabel="Perda (MSE/2)", yscale="log"); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------- figuras de cáusticas (banner v3)
def _vbm():
    import VBMicrolensing
    V = VBMicrolensing.VBMicrolensing(); V.Tol = 1e-3; return V


def _track(V, s, q, u0, alpha, tE, t0, rho=None, tspan=1.6):
    t = np.linspace(t0 - tspan * tE, t0 + tspan * tE, 4000)
    A, y1, y2 = V.BinaryLightCurve([np.log(s), np.log(q), u0, alpha, np.log(rho or C.RHO), np.log(tE), t0], t)
    return t, np.asarray(A), np.asarray(y1), np.asarray(y2)


def _lens_xy(s, q):   # origem no centro de massa (convenção VBM)
    return -s * q / (1 + q), s / (1 + q)


def through_caustic(V, s, q, which="right", alpha=0.5):
    """(u0, alpha) da reta que passa pelo centro da cáustica mais à esquerda/direita (convenção VBM)."""
    cs = V.Caustics(s, q); m = [(np.mean(c[0]), np.mean(c[1])) for c in cs]
    k = int(np.argmin([x for x, _ in m])) if which == "left" else int(np.argmax([x for x, _ in m]))
    px, py = m[k]; u0 = px * np.sin(alpha) - py * np.cos(alpha)
    return float(abs(u0)), float(alpha if u0 >= 0 else (alpha + np.pi) % (2 * np.pi))


def fig_topologies(path, q=1e-2):
    """Close / resonante / wide: cáustica + trajetória (cima) e curva de luz correspondente (baixo)."""
    V = _vbm()
    def through(s, which, alpha):                 # reta que passa pelo centro da cáustica escolhida
        cs = V.Caustics(s, q); m = [(np.mean(c[0]), np.mean(c[1])) for c in cs]
        k = int(np.argmin([x for x, _ in m])) if which == "left" else int(np.argmax([x for x, _ in m]))
        px, py = m[k]; u0 = px * np.sin(alpha) - py * np.cos(alpha)
        return (abs(u0), alpha if u0 >= 0 else (alpha + np.pi) % (2 * np.pi))
    cases = [("Close (s < 1)", 0.6) + through(0.6, "left", 2.0), ("Ressonante (s ≈ 1)", 1.0, 0.10, 0.7), ("Wide (s > 1)", 1.6) + through(1.6, "right", 0.5)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4), gridspec_kw={"height_ratios": [1.25, 1]})
    for j, (name, s, u0, al) in enumerate(cases):
        t, A, y1, y2 = _track(V, s, q, u0, al, 30.0, 0.0); a, b = axs[0, j], axs[1, j]
        for c in V.Caustics(s, q): a.plot(c[0], c[1], color="#d62728", lw=2.2)
        xs, xp = _lens_xy(s, q); a.plot(xs, 0, "o", color="k", ms=11); a.plot(xp, 0, "o", color="#ff7f0e", ms=7)
        a.plot(y1, y2, color="#1f4e79", lw=1.6); k = len(y1) // 2 + 250
        a.annotate("", xy=(y1[k + 30], y2[k + 30]), xytext=(y1[k], y2[k]), arrowprops=dict(arrowstyle="-|>", color="#1f4e79", lw=2))
        cx = np.concatenate([np.asarray(c[0]) for c in V.Caustics(s, q)]); cy = np.concatenate([np.asarray(c[1]) for c in V.Caustics(s, q)])
        a.set_xlim(min(cx.min(), xs) - 0.35, max(cx.max(), xp) + 0.35); a.set_ylim(-0.85, 0.85); a.set_aspect("equal")
        a.set_title(f"{name}\ns = {s}, q = {q:g}", fontsize=17); a.set_xlabel("x (θ$_E$)")
        if j == 0: a.set_ylabel("y (θ$_E$)")
        b.plot(t, A, color="k", lw=1.6); b.set_xlabel("Tempo (dias)"); b.set_yscale("log")
        if j == 0: b.set_ylabel("Amplificação")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_event(p, data, fit, path):
    """Evento 2L1S observado: (esq.) cáustica + trajetória; (dir.) dados + ajuste 1L1S, e resíduo = entrada da rede."""
    V = _vbm(); s, q = 10 ** p["logs"], 10 ** p["logq"]
    t, A, y1, y2 = _track(V, s, q, p["u0"], p["alpha"], p["tE"], p["t0"])
    fig = plt.figure(figsize=(14, 6.4)); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6], height_ratios=[2.2, 1], hspace=0.08, wspace=0.22)
    a = fig.add_subplot(gs[:, 0]); b = fig.add_subplot(gs[0, 1]); r = fig.add_subplot(gs[1, 1], sharex=b)
    cs = V.Caustics(s, q)
    for c in cs: a.plot(c[0], c[1], color="#d62728", lw=2.2)
    xs, xp = _lens_xy(s, q); a.plot(xs, 0, "o", color="k", ms=11); a.plot(xp, 0, "o", color="#ff7f0e", ms=7); a.plot(y1, y2, color="#1f4e79", lw=1.6)
    cx = np.concatenate([np.asarray(c[0]) for c in cs]); cy = np.concatenate([np.asarray(c[1]) for c in cs])
    a.set_xlim(min(cx.min(), xs) - 0.5, max(cx.max(), xp) + 0.5); a.set_ylim(-0.75, 0.75)
    a.set_aspect("equal"); a.set(xlabel="x (θ$_E$)", ylabel="y (θ$_E$)")
    a.set_title(f"s = {s:.2f}, q = {q:.1e}, α = {np.degrees(p['alpha']):.0f}°", fontsize=16)
    tt = np.concatenate([d[0] for d in data.values()])
    for name, (tk, f, e) in data.items(): b.errorbar(tk, f, e, fmt=".", ms=3, lw=0.4, alpha=0.55, color=COL[name], label=name)
    b.plot(t, A, color="k", lw=1.2, label="2L1S (verdadeiro)"); o = np.argsort(tt)
    from .features import _A
    td = np.linspace(tt.min(), tt.max(), 3000); b.plot(td, fit["fb"] + fit["fs"] * _A(td, fit["t0"], fit["u0"], fit["tE"]), color="#ff7f0e", lw=2, label="melhor 1L1S")
    b.set_ylabel("Amplificação"); b.legend(ncol=1, fontsize=12, loc="upper right")
    plt.setp(b.get_xticklabels(), visible=False)
    res = (np.concatenate([d[1] for d in data.values()]) - fit["model"]) / np.concatenate([d[2] for d in data.values()])
    r.plot(tt[o], res[o], ".", ms=3, color="#444444", alpha=.7); r.axhline(0, color="#ff7f0e"); r.set(xlabel="Tempo (dias)", ylabel="resíduo (σ)")
    lo, hi = p["t0"] - 1.3 * p["tE"], p["t0"] + 1.3 * p["tE"]; b.set_xlim(lo, hi); fig.savefig(path); plt.close(fig)


def fig_pred_true_cnn(y, P, d2, path, thr=100.0):
    """Previsto × verdadeiro da CNN no subconjunto Δχ² > thr. P: dict de mlens.torch_cnn.decode."""
    from sklearn.metrics import r2_score
    m = d2 > thr; yt = y[m]; at = np.arctan2(yt[:, 6], yt[:, 5]) % (2 * np.pi)
    items = [("t$_0$ (d)", yt[:, 0], P["t0"][m]), ("u$_0$", yt[:, 1], P["u0"][m]), ("t$_E$ (d)", yt[:, 2], P["tE"][m]),
             ("log q", yt[:, 3], P["logq"][m]), ("|log s|", np.abs(yt[:, 4]), P["abs_logs"][m])]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.2)); axs = axs.ravel()
    for ax, (lab, a, b) in zip(axs, items):
        ax.scatter(a, b, s=5, alpha=0.35, color="#1f4e79", rasterized=True); lo, hi = a.min(), a.max(); ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set(xlabel=f"{lab} verdadeiro", ylabel="previsto"); ax.set_title(f"R² = {r2_score(a, b):.2f}")
    ax = axs[5]; ap = P["alpha"][m]; ax.scatter(at, ap, s=5, alpha=0.35, color="#8b1a1a", rasterized=True); ax.plot([0, 2*np.pi], [0, 2*np.pi], "k--", lw=1.2)
    e = np.degrees(np.abs(np.angle(np.exp(1j * (ap - at))))).mean(); ax.set(xlabel="α verdadeiro (rad)", ylabel="previsto"); ax.set_title(f"erro médio = {e:.0f}°")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
