"""Métricas de ERRO por parâmetro (numpy). O R² fica só como informação secundária.
Convenções: erro = previsto - verdadeiro. α usa erro angular com wrap em (-180°, 180°]. O sinal de log s é tratado como classificação (s > 1)."""
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

EXC_EDGES = [-np.inf, 10, 100, 1000, np.inf]
EXC_LABELS = ["<10", "10-100", "100-1000", ">1000"]      # excesso de chi2 do ajuste 1L1S (chi2 - N)


def wrap_deg(a, b):
    return np.degrees(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


def _stats(e):
    e = np.asarray(e, float)
    return dict(bias=float(np.median(e)), med_abs=float(np.median(np.abs(e))),
                sigma68=float(0.5 * (np.percentile(e, 84) - np.percentile(e, 16))), rmse=float(np.sqrt(np.mean(e ** 2))))


def alpha_true(y):
    return np.arctan2(y[:, 6], y[:, 5]) % (2 * np.pi)


def summary(y, P, m=None):
    m = np.ones(len(y), bool) if m is None else m
    out = {"n": int(m.sum())}
    if out["n"] < 20:
        return out
    y = y[m]; P = {k: np.asarray(v)[m] for k, v in P.items()}
    err = {"t0": P["t0"] - y[:, 0], "u0": P["u0"] - y[:, 1], "tE_pct": 100 * (P["tE"] - y[:, 2]) / y[:, 2],
           "logq": P["logq"] - y[:, 3], "logs": P["logs"] - y[:, 4], "abs_logs": P["abs_logs"] - np.abs(y[:, 4]),
           "alpha_deg": wrap_deg(P["alpha"], alpha_true(y))}
    for k, e in err.items():
        out[k] = _stats(e)
    out["alpha_deg"]["frac_lt30"] = float(np.mean(np.abs(err["alpha_deg"]) < 30))
    out["r2"] = {k: float(r2_score(a, b)) for k, a, b in [("t0", y[:, 0], P["t0"]), ("u0", y[:, 1], P["u0"]), ("tE", y[:, 2], P["tE"]),
                 ("logq", y[:, 3], P["logq"]), ("abs_logs", np.abs(y[:, 4]), P["abs_logs"])]}
    yt, yp = y[:, 4] > 0, P["p_wide"] > 0.5
    rp = float(np.mean(yp[yt])) if yt.any() else np.nan
    rn = float(np.mean(~yp[~yt])) if (~yt).any() else np.nan
    nr = np.abs(y[:, 4]) > 0.15                              # fora da região ressonante (s próximo de 1)
    out["sign"] = dict(acc=float(np.mean(yt == yp)), bal_acc=float(np.nanmean([rp, rn])), recall_wide=rp, recall_close=rn,
                       auc=float(roc_auc_score(yt, P["p_wide"])) if (yt.any() and (~yt).any()) else float("nan"),
                       acc_nonres=float(np.mean(yt[nr] == yp[nr])) if nr.sum() > 20 else float("nan"), frac_nonres=float(nr.mean()))
    return out


def full_report(y, P, excess):
    rep = {"all": summary(y, P), "exc>100": summary(y, P, excess > 100), "bins": {}}
    for lab, lo, hi in zip(EXC_LABELS, EXC_EDGES[:-1], EXC_EDGES[1:]):
        rep["bins"][lab] = summary(y, P, (excess > lo) & (excess <= hi))
    return rep


def print_report(rep, title=""):
    print(f"\n== {title} ==")
    for key in ("all", "exc>100"):
        s = rep[key]
        if s["n"] < 20:
            continue
        print(f"[{key}] n={s['n']}  erro absoluto mediano / sigma68:  t0 {s['t0']['med_abs']:.2f}/{s['t0']['sigma68']:.2f} d | u0 {s['u0']['med_abs']:.3f}/{s['u0']['sigma68']:.3f} | "
              f"tE {s['tE_pct']['med_abs']:.1f}/{s['tE_pct']['sigma68']:.1f} % | logq {s['logq']['med_abs']:.2f}/{s['logq']['sigma68']:.2f} dex | "
              f"logs {s['logs']['med_abs']:.2f}/{s['logs']['sigma68']:.2f} dex | alpha {s['alpha_deg']['med_abs']:.0f}/{s['alpha_deg']['sigma68']:.0f} graus "
              f"(<30: {100 * s['alpha_deg']['frac_lt30']:.0f}%) | sinal de log s: acerto {100 * s['sign']['acc']:.0f}%, balanceado {100 * s['sign']['bal_acc']:.0f}%, AUC {s['sign']['auc']:.2f}")
