"""Avaliação em eventos de LENTE SIMPLES (1L1S): a rede de detecção classifica e estima t0, u0, tE.
Não retreina nada.

Versão final (tópico 4 da revisão): o limiar é escolhido numa amostra de CALIBRAÇÃO sem planeta (--null-calib, a mesma
usada pelo 10_deteccao.py) e a fração classificada como lente simples é medida numa amostra INDEPENDENTE (--null-eval),
gerada com outra semente. Na versão anterior as duas eram o mesmo arquivo, e os 95% eram consequência da definição do limiar.

Compara a rede com o ajuste 1L1S por mínimos quadrados (que já está nos escalares 'sc' de cada evento).
Uso (na pasta do projeto):   python scripts/11_avaliar_1l1s.py --run final_det
Saídas: models/avaliacao_1l1s_<run>.json, figures/fig_1l1s_<run>.png e, no terminal, as linhas da tabela em LaTeX."""
import sys, json, pathlib, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mlens.torch_cnn import predict, pspl_from_sc

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="final_det")
ap.add_argument("--null-calib", default=str(ROOT / "data/processed/test_null.npz"), help="sem planeta: só para escolher o limiar")
ap.add_argument("--null-eval", default=str(ROOT / "data/processed/test_null_indep.npz"), help="sem planeta, independente: avaliação")
ap.add_argument("--fpr", type=float, default=0.05); a = ap.parse_args()
model = str(ROOT / f"models/cnn_{a.run}.pt")
pc = predict(model, np.load(a.null_calib)["X"], np.load(a.null_calib)["sc"])["p_planet"]
lim = float(np.quantile(pc, 1 - a.fpr))                                # limiar: fpr dos eventos de calibração acima dele
z = np.load(a.null_eval); y, sc = z["y"], z["sc"]
if len(y) == len(np.load(a.null_calib)["y"]) and np.allclose(y, np.load(a.null_calib)["y"]):
    sys.exit("ERRO: --null-eval é igual a --null-calib; gere uma amostra independente (ver reproduzir.py).")
P = predict(model, z["X"], sc)
simples = P["p_planet"] < lim                                          # classificados como lente simples
k_, n_ = int(simples.sum()), len(simples)
_p = k_ / n_; _c = (_p + 1.92 / n_) / (1 + 3.84 / n_); _h = 1.96 * np.sqrt(_p * (1 - _p) / n_ + 0.96 / n_ ** 2) / (1 + 3.84 / n_)

# ajuste 1L1S: sc = [log10 tE, log10 u0, fs, t0 normalizado na janela]
_t0, _u0, _tE = pspl_from_sc(sc); fit = dict(t0=_t0, u0=_u0, tE=_tE)
rede = dict(t0=P["t0"], u0=P["u0"], tE=P["tE"])
verd = dict(t0=y[:, 0], u0=y[:, 1], tE=y[:, 2])


def r2(t, p): return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def erros(est, m):
    return dict(R2={k: r2(verd[k][m], est[k][m]) for k in verd},
                t0_med_dias=float(np.median(np.abs(est["t0"][m] - verd["t0"][m]))),
                u0_med=float(np.median(np.abs(est["u0"][m] - verd["u0"][m]))),
                tE_med_pct=float(100 * np.median(np.abs(est["tE"][m] / verd["tE"][m] - 1))))


todos = np.ones(len(y), bool)
R = {"n": int(len(y)), "arquivo_calibracao": a.null_calib, "arquivo_avaliacao": a.null_eval, "n_calibracao": int(len(pc)),
     "limiar": lim, "fpr_alvo": a.fpr, "frac_classificados_simples": _p, "ic95_frac_classificados_simples": [float(_c - _h), float(_c + _h)],
     "rede_todos": erros(rede, todos), "rede_classificados_simples": erros(rede, simples), "ajuste_1L1S_todos": erros(fit, todos)}
(ROOT / "models").mkdir(exist_ok=True)
json.dump(R, open(ROOT / f"models/avaliacao_1l1s_{a.run}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
print(f"\nClassificados como lente simples na amostra independente: {100 * _p:.1f}% (IC 95%: {100 * R['ic95_frac_classificados_simples'][0]:.1f}–{100 * R['ic95_frac_classificados_simples'][1]:.1f}%)")

print("\n=== Linhas para a tabela LaTeX (R² | erro mediano) ===")
f = lambda x, d=3: f"{x:.{d}f}".replace(".", "{,}")
for nome, chave in (("Rede (todos)", "rede_todos"), ("Rede (classificados 1L1S)", "rede_classificados_simples"), ("Ajuste 1L1S", "ajuste_1L1S_todos")):
    e = R[chave]
    print(f"{nome} & {f(e['R2']['t0'])} & {f(e['R2']['u0'])} & {f(e['R2']['tE'])} & "
          f"{f(e['t0_med_dias'], 2)} & {f(e['u0_med'])} & {f(e['tE_med_pct'], 1)} \\\\")

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for a_, k, lab in zip(ax, ("t0", "u0", "tE"), (r"$t_0$ (dias)", r"$u_0$", r"$t_\mathrm{E}$ (dias)")):
    a_.scatter(verd[k], rede[k], s=3, alpha=.5, color="#3b6ea5", rasterized=True)
    lo, hi = verd[k].min(), verd[k].max(); a_.plot([lo, hi], [lo, hi], "k--", lw=1)
    a_.set_xlabel(lab + " verdadeiro"); a_.set_ylabel(lab + " previsto"); a_.set_title(f"$R^2$ = {R['rede_todos']['R2'][k]:.2f}")
plt.tight_layout(); (ROOT / "figures").mkdir(exist_ok=True)
plt.savefig(ROOT / f"figures/fig_1l1s_{a.run}.png", dpi=170); print(f"salvo: figures/fig_1l1s_{a.run}.png")
