"""R² de TODOS os parâmetros, sem retreinar. Lê as previsões salvas por 02b_treinar_cnn.py.
Uso:   python scripts/07_r2.py                                  (padrão: teste com priori uniforme)
       python scripts/07_r2.py --pred data/processed/test_pred_mix.npz --tag mix
Saídas: models/r2_<tag>.json  e  figures/fig_r2_<tag>.png (previsto x verdadeiro)

Convenções (iguais a mlens/metrics.py): y = [t0, u0, tE, log q, log s, cos α, sin α].
- tE: R² em log10(tE) (coerente com erro relativo em %); o R² linear também é salvo.
- log s: R² com sinal; R² de |log s| (cabeça própria) e de |saída com sinal| — ambos contra |log s| verdadeiro.
- α: R² circular = 1 - Σ|e^{iα̂} - e^{iα}|² / Σ|e^{iα} - média|² ≈ 2<cos Δα> - 1.
  ATENÇÃO: chute aleatório dá ≈ -1 (não 0); R² = 0 equivale a <cos Δα> = 0,5. Ver também fração com erro < 30° (acaso: 16,7%).
IC de 95% por bootstrap (reamostragem dos eventos)."""
import sys, pathlib, json, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for _s in (sys.stdout, sys.stderr):            # console do Windows (cp1252) não imprime α, χ², etc.
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--pred", default=str(ROOT / "data/processed/test_pred_uniform.npz"))
ap.add_argument("--tag", default="uniform")
ap.add_argument("--boot", type=int, default=1000, help="reamostragens do bootstrap (0 desliga)")
ap.add_argument("--corte", type=float, default=100.0, help="corte no excesso de χ² do ajuste 1L1S")
a = ap.parse_args()

d = np.load(a.pred)
y, ex = d["y"], d["d2"]
alpha_p = np.asarray(d["alpha"], float)
if np.abs(alpha_p).max() > 4 * np.pi:       # salvo em graus -> radianos
    alpha_p = np.radians(alpha_p)
alpha_t = np.arctan2(y[:, 6], y[:, 5])


def r2(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    return 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)


def r2_circ(t, p):
    zt, zp = np.exp(1j * t), np.exp(1j * p)
    return 1 - np.sum(np.abs(zp - zt) ** 2) / np.sum(np.abs(zt - zt.mean()) ** 2)


# nome -> (verdadeiro, previsto, função)
def pares(m):
    lt, lp = y[m, 4], d["logs"][m]
    return {
        "t0":            (y[m, 0], d["t0"][m], r2),
        "u0":            (y[m, 1], d["u0"][m], r2),
        "log tE":        (np.log10(y[m, 2]), np.log10(np.clip(d["tE"][m], 1e-3, None)), r2),
        "tE (linear)":   (y[m, 2], d["tE"][m], r2),
        "log q":         (y[m, 3], d["logq"][m], r2),
        "log s":         (lt, lp, r2),
        "|log s|":       (np.abs(lt), d["abs_logs"][m], r2),
        "|saída log s|": (np.abs(lt), np.abs(lp), r2),
        "α (circular)":  (alpha_t[m], alpha_p[m], r2_circ),
    }


rng = np.random.default_rng(0)


def avalia(m):
    out = {"n": int(m.sum())}
    for nome, (t, p, f) in pares(m).items():
        v = float(f(t, p)); res = {"r2": v}
        if a.boot:
            n = len(t); bs = [f(t[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(a.boot))]
            res["ic95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[nome] = res
    err = np.degrees(np.abs(np.angle(np.exp(1j * (alpha_p[m] - alpha_t[m])))))
    out["α: fração erro<30°"] = float(np.mean(err < 30))
    out["α: <cos Δα>"] = float(np.mean(np.cos(np.radians(err))))
    return out


grupos = {"todos": np.ones(len(y), bool), f"excesso>{a.corte:g}": ex > a.corte}
rep = {g: avalia(m) for g, m in grupos.items()}

# --- tabela no terminal ---
nomes = list(pares(grupos["todos"]).keys())
cab = f"{'parâmetro':<18}" + "".join(f"{g + ' (n=' + str(rep[g]['n']) + ')':>30}" for g in rep)
print(f"\nR² — {a.pred}\n" + cab + "\n" + "-" * len(cab))
for nm in nomes:
    linha = f"{nm:<18}"
    for g in rep:
        r = rep[g][nm]; ic = f" [{r['ic95'][0]:.2f}; {r['ic95'][1]:.2f}]" if "ic95" in r else ""
        linha += f"{r['r2']:>12.3f}{ic:>18}"
    print(linha)
print(f"{'α: erro<30°':<18}" + "".join(f"{100 * rep[g]['α: fração erro<30°']:>29.0f}%" for g in rep) + "   (acaso: 17%)")
print(f"{'α: <cos Δα>':<18}" + "".join(f"{rep[g]['α: <cos Δα>']:>30.2f}" for g in rep) + "   (acaso: 0)")

(ROOT / "models").mkdir(exist_ok=True)
json.dump(rep, open(ROOT / f"models/r2_{a.tag}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# --- figura: previsto x verdadeiro (cor = excesso de χ²) ---
pp = pares(grupos["todos"])
paineis = [("t0", "t$_0$ (dias)"), ("u0", "u$_0$"), ("log tE", r"$\log_{10} t_E$"),
           ("log q", r"$\log_{10} q$"), ("log s", r"$\log_{10} s$"), ("α (circular)", r"$\alpha$ (graus)")]
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
cor = np.log10(np.clip(ex, 1, None))
for A, (k, lab) in zip(ax.ravel(), paineis):
    t, p, _ = pp[k]
    if k.startswith("α"):
        t, p = np.degrees(t) % 360, np.degrees(p) % 360
    sc = A.scatter(t, p, c=cor, s=3, alpha=.5, cmap="viridis", rasterized=True)
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    A.plot([lo, hi], [lo, hi], "k--", lw=1)
    if k == "log s":
        A.plot([lo, hi], [-lo, -hi], ":", color="gray", lw=1)    # ramo degenerado s -> 1/s
    A.set_xlabel(lab + " verdadeiro"); A.set_ylabel(lab + " previsto")
    A.set_title(f"R² = {rep['todos'][k]['r2']:.2f}", fontsize=13)
fig.colorbar(sc, ax=ax, shrink=.8, label=r"$\log_{10}$ (excesso de $\chi^2$ do ajuste 1L1S)")
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / f"figures/fig_r2_{a.tag}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"\nsalvo: models/r2_{a.tag}.json  e  figures/fig_r2_{a.tag}.png")
