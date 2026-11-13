"""A rede CORRIGE o ajuste de lente simples (PSPL) ou só o REPRODUZ?  (tópico 2 da revisão da banca)

Compara, evento a evento e no MESMO conjunto de teste 2L1S, o erro de t0, u0 e tE de duas estimativas:
  - PSPL: o ajuste de lente simples por mínimos quadrados, que a rede recebe como entrada (escalares 'sc');
  - rede: a previsão da CNN.
Não retreina nada. Funciona com qualquer rede (versões antigas, que preveem o valor absoluto, ou a final, com --delta).

Medidas, por faixa de excesso de χ² do ajuste PSPL:
  1. erro absoluto mediano de cada estimador (IC de 95% por bootstrap) e a razão rede/PSPL;
  2. fração dos eventos em que a rede chega mais perto do valor verdadeiro que o PSPL (IC de Wilson)
     e o teste de postos sinalizados de Wilcoxon para |e_PSPL| − |e_rede| (pareado, bilateral);
  3. FRAÇÃO DA CORREÇÃO APLICADA, β. Para cada evento:
         correção necessária   c = verdadeiro − PSPL
         correção aplicada     a = rede − PSPL
     β é a inclinação da reta a = β c (mínimos quadrados pela origem): β = Σ a c / Σ c².
     β ≈ 0: a rede devolve o que recebeu (reproduz);  β ≈ 1: a rede aplica toda a correção necessária.
     Também se reporta a correlação de Pearson entre a e c.
     β e a correlação usam só os eventos em que o ajuste PSPL ficou numa faixa física (mlens.torch_cnn.pspl_fisico); nos demais
     (~8% do teste uniforme) o PSPL diverge (ex.: tE ~ 1e9 d) e um único evento dominaria a soma de quadrados. Esses eventos
     são reportados à parte: n e erro mediano do PSPL e da rede.
Variáveis: t0 em dias; u0 linear; tE em log10 (erro relativo).

Uso:   python scripts/12_rede_vs_pspl.py --run final
       python scripts/12_rede_vs_pspl.py --pred data/processed/test_pred_uniform_v2b.npz --test data/processed/test_uniform.npz --tag v2b
Saídas: models/rede_vs_pspl_<tag>.json, figures/fig_rede_vs_pspl_<tag>.png e as linhas da tabela LaTeX no terminal."""
import sys, json, pathlib, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mlens.torch_cnn import pspl_from_sc, pspl_fisico

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="final", help="lê data/processed/test_pred_uniform_<run>.npz")
ap.add_argument("--pred", default=None, help="arquivo de previsões (sobrepõe --run)")
ap.add_argument("--test", default=str(ROOT / "data/processed/test_uniform.npz"), help="conjunto de teste (só usado se o arquivo de previsões não tiver 'sc')")
ap.add_argument("--tag", default=None)
ap.add_argument("--logq-max", type=float, default=None, help="se dado, usa só eventos com log10 q < valor (ex.: -1.52 para q < 0,03)")
ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
pred = pathlib.Path(a.pred) if a.pred else ROOT / f"data/processed/test_pred_uniform_{a.run}.npz"
tag = a.tag or (a.run if not a.pred else pred.stem.replace("test_pred_", ""))
if a.logq_max is not None:
    tag += f"_logq<{a.logq_max:g}"

d = np.load(pred); y, ex = d["y"], d["d2"]
if "sc" in d.files:
    sc = d["sc"]
else:                                                              # previsões antigas: escalares vêm do conjunto de teste
    t = np.load(a.test)
    if len(t["y"]) != len(y) or not np.allclose(t["y"], y):
        sys.exit(f"ERRO: {a.test} não corresponde a {pred} (eventos diferentes ou em outra ordem).")
    sc = t["sc"]
if "planet" in d.files:                                            # rede de detecção: só eventos com planeta
    k = d["planet"] > 0; y, ex, sc = y[k], ex[k], sc[k]; P = {kk: np.asarray(d[kk])[k] for kk in ("t0", "u0", "tE")}
else:
    P = {kk: np.asarray(d[kk]) for kk in ("t0", "u0", "tE")}
if a.logq_max is not None:
    k = y[:, 3] < a.logq_max; y, ex, sc = y[k], ex[k], sc[k]; P = {kk: v[k] for kk, v in P.items()}
t0f, u0f, tEf = pspl_from_sc(sc); FIS = pspl_fisico(sc)

lg = lambda v: np.log10(np.clip(np.asarray(v, float), 1e-6, None))
# (verdadeiro, PSPL, rede) na variável em que o erro é medido
VAR = {"t0": (y[:, 0], t0f, P["t0"]), "u0": (y[:, 1], u0f, P["u0"]), "tE": (lg(y[:, 2]), lg(tEf), lg(P["tE"]))}
UNID = {"t0": "dias", "u0": "", "tE": "dex (log10 tE)"}
EDG = [-np.inf, 10, 100, 1000, np.inf]; LAB = ["<10", "10-100", "100-1000", ">1000"]
GRUPOS = {"todos": np.ones(len(y), bool), "excesso>100": ex > 100}
GRUPOS.update({f"excesso {l}": (ex > lo) & (ex <= hi) for l, lo, hi in zip(LAB, EDG[:-1], EDG[1:])})
rng = np.random.default_rng(a.seed)


def wilson(k, n, z=1.96):
    if n == 0: return [float("nan")] * 2
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def boot(f, *arrs):
    if not a.boot: return [float("nan")] * 2
    n = len(arrs[0]); v = [f(*(x[i] for x in arrs)) for i in (rng.integers(0, n, n) for _ in range(a.boot))]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def beta(ap_, c):
    den = np.sum(c * c); return float(np.sum(ap_ * c) / den) if den > 0 else float("nan")


def analisa(m):
    out = {"n": int(m.sum()), "n_pspl_nao_fisico": int((m & ~FIS).sum())}
    if out["n"] < 20: return out
    for nome, (v, f, r) in VAR.items():
        fis = FIS[m]; v, f, r = v[m], f[m], r[m]
        ef, er = np.abs(f - v), np.abs(r - v)                      # erros absolutos
        c, ap_ = (v - f)[fis], (r - f)[fis]                        # correção necessária e aplicada (PSPL físico)
        melhor = int(np.sum(er < ef))
        try: pw = float(wilcoxon(ef, er).pvalue)
        except ValueError: pw = float("nan")
        out[nome] = dict(
            erro_mediano_pspl=float(np.median(ef)), ic95_pspl=boot(np.median, ef),
            erro_mediano_rede=float(np.median(er)), ic95_rede=boot(np.median, er),
            razao_rede_pspl=float(np.median(er) / np.median(ef)) if np.median(ef) > 0 else float("nan"),
            ic95_razao=boot(lambda x, z: np.median(z) / np.median(x) if np.median(x) > 0 else np.nan, ef, er),
            frac_rede_melhor=melhor / len(v), ic95_frac=wilson(melhor, len(v)), p_wilcoxon=pw,
            beta=beta(ap_, c), ic95_beta=boot(beta, ap_, c),
            corr=float(np.corrcoef(ap_, c)[0, 1]) if np.std(ap_) > 0 and np.std(c) > 0 else float("nan"),
            correcao_necessaria_mediana=float(np.median(np.abs(c))), correcao_aplicada_mediana=float(np.median(np.abs(ap_))),
            pspl_nao_fisico=dict(n=int((~fis).sum()), erro_mediano_pspl=float(np.median(ef[~fis])) if (~fis).any() else None,
                                 erro_mediano_rede=float(np.median(er[~fis])) if (~fis).any() else None))
    return out


R = {"arquivo": str(pred), "variaveis": UNID, "grupos": {g: analisa(m) for g, m in GRUPOS.items()}}
(ROOT / "models").mkdir(exist_ok=True)
json.dump(R, open(ROOT / f"models/rede_vs_pspl_{tag}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# ---------------- terminal ----------------
print(f"\nRede x ajuste PSPL — {pred}")
for g, r in R["grupos"].items():
    if r["n"] < 20: continue
    print(f"\n[{g}]  n = {r['n']}   (ajuste PSPL fora da faixa física: {r['n_pspl_nao_fisico']})")
    print(f"  {'parâm.':<6}{'erro PSPL':>12}{'erro rede':>12}{'razão':>8}{'rede melhor':>14}{'p Wilcoxon':>13}{'β (IC 95%)':>22}{'corr':>7}")
    for k, s in ((k, r[k]) for k in VAR):
        print(f"  {k:<6}{s['erro_mediano_pspl']:>12.4g}{s['erro_mediano_rede']:>12.4g}{s['razao_rede_pspl']:>8.2f}"
              f"{100 * s['frac_rede_melhor']:>13.0f}%{s['p_wilcoxon']:>13.1e}"
              f"{s['beta']:>9.2f} [{s['ic95_beta'][0]:.2f}; {s['ic95_beta'][1]:.2f}]{s['corr']:>7.2f}")
s_ = R["grupos"]["todos"]
if s_["n_pspl_nao_fisico"]:
    print(f"\nEventos com ajuste PSPL divergente (n = {s_['n_pspl_nao_fisico']}): erro mediano PSPL -> rede: "
          + "  ".join(f"{k} {s_[k]['pspl_nao_fisico']['erro_mediano_pspl']:.3g} -> {s_[k]['pspl_nao_fisico']['erro_mediano_rede']:.3g}" for k in VAR))
print("\nβ ≈ 0: a rede reproduz o ajuste PSPL;  β ≈ 1: a rede aplica toda a correção necessária.  Unidades:", UNID)

print("\n=== Linhas para a tabela LaTeX: erro mediano PSPL | rede | rede melhor (%) | β — todos e excesso > 100 ===")
fmt = lambda x, n=2: f"{x:.{n}f}".replace(".", "{,}")
nomes = {"t0": r"$t_0$ (dias)", "u0": r"$u_0$", "tE": r"$\log t_\mathrm{E}$ (dex)"}
for k in VAR:
    cel = []
    for g in ("todos", "excesso>100"):
        s = R["grupos"][g][k]; n = 3 if k != "t0" else 2
        cel.append(f"{fmt(s['erro_mediano_pspl'], n)} & {fmt(s['erro_mediano_rede'], n)} & {100 * s['frac_rede_melhor']:.0f} & {fmt(s['beta'])}")
    print(f"{nomes[k]} & " + " & ".join(cel) + r" \\")

# ---------------- figura ----------------
fig, ax = plt.subplots(2, 3, figsize=(15, 10)); fig.subplots_adjust(hspace=0.38, wspace=0.3)
x = np.arange(len(LAB)); cor = np.log10(np.clip(ex, 1, None))
TIT = {"t0": r"$t_0$ (dias)", "u0": r"$u_0$", "tE": r"$\log_{10} t_\mathrm{E}$"}
for j, k in enumerate(VAR):
    A = ax[0, j]
    for est, col, dx, rot in (("pspl", "#e67e22", -.1, "ajuste PSPL"), ("rede", "#3b6ea5", .1, "rede")):
        md = [R["grupos"][f"excesso {l}"][k][f"erro_mediano_{est}"] for l in LAB]
        ic = np.array([R["grupos"][f"excesso {l}"][k][f"ic95_{est}"] for l in LAB])
        A.errorbar(x + dx, md, yerr=[np.array(md) - ic[:, 0], ic[:, 1] - np.array(md)], fmt="o-", color=col, capsize=4, label=rot)
    A.set_yscale("log"); A.set_xticks(x, LAB); A.set_xlabel(r"excesso de $\chi^2$ do ajuste PSPL")
    A.set_ylabel("erro absoluto mediano"); A.set_title(f"({'abc'[j]}) {TIT[k]}"); A.legend(fontsize=9)
    v, f, r = VAR[k]; c, ap_ = (v - f)[FIS], (r - f)[FIS]
    lim = np.nanpercentile(np.abs(np.r_[c, ap_]), 99) * 1.1
    B = ax[1, j]; sc_ = B.scatter(c, ap_, c=cor[FIS], s=3, alpha=.5, cmap="viridis", rasterized=True)
    B.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="correção total (β = 1)"); B.axhline(0, color="gray", ls=":", lw=1, label="só reproduz (β = 0)")
    B.set_xlim(-lim, lim); B.set_ylim(-lim, lim); B.legend(fontsize=8, loc="upper left")
    B.set_xlabel("correção necessária (verdadeiro − PSPL)"); B.set_ylabel("correção aplicada (rede − PSPL)")
    B.set_title(f"({'def'[j]}) {TIT[k]}:  β = {R['grupos']['todos'][k]['beta']:.2f}")
fig.colorbar(sc_, ax=ax[1, :], shrink=.8, label=r"$\log_{10}$ (excesso de $\chi^2$)")
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / f"figures/fig_rede_vs_pspl_{tag}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"\nsalvo: models/rede_vs_pspl_{tag}.json  e  figures/fig_rede_vs_pspl_{tag}.png")
