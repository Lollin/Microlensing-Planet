"""Avaliação em eventos de LENTE SIMPLES (1L1S): a rede de detecção classifica e estima t0, u0, tE.
Não retreina nada. Usa:
  models/cnn_v3.pt                 (rede de detecção, com a saída de probabilidade de planeta)
  models/deteccao_v3.json          (limiar com 5% de falsos positivos, gerado pelo 10_deteccao.py)
  data/processed/test_null.npz     (3000 eventos sem planeta)
Compara a rede com o ajuste 1L1S por mínimos quadrados (que já está nos escalares 'sc' de cada evento).

Uso (na pasta do projeto):   python scripts/11_avaliar_1l1s.py
Saídas: models/avaliacao_1l1s.json, figures/fig_1l1s.png e, no terminal, as linhas da tabela em LaTeX."""
import sys, json, pathlib, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mlens.torch_cnn import predict
from mlens import config as C

z = np.load(ROOT / "data/processed/test_null.npz"); y, sc = z["y"], z["sc"]
P = predict(str(ROOT / "models/cnn_v3.pt"), z["X"], sc)
lim = json.load(open(ROOT / "models/deteccao_v3.json", encoding="utf-8"))["por_taxa_falso_positivo"]["5%"]["rede"]["limiar"]
simples = P["p_planet"] < lim                                          # classificados como lente simples

# ajuste 1L1S: sc = [log10 tE, log10 u0, fs, t0 normalizado na janela]
fit = dict(t0=C.T_MIN + sc[:, 3] * (C.T_MAX - C.T_MIN), u0=10 ** sc[:, 1], tE=10 ** sc[:, 0])
rede = dict(t0=P["t0"], u0=P["u0"], tE=P["tE"])
verd = dict(t0=y[:, 0], u0=y[:, 1], tE=y[:, 2])


def r2(t, p): return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def erros(est, m):
    return dict(R2={k: r2(verd[k][m], est[k][m]) for k in verd},
                t0_med_dias=float(np.median(np.abs(est["t0"][m] - verd["t0"][m]))),
                u0_med=float(np.median(np.abs(est["u0"][m] - verd["u0"][m]))),
                tE_med_pct=float(100 * np.median(np.abs(est["tE"][m] / verd["tE"][m] - 1))))


todos = np.ones(len(y), bool)
R = {"n": int(len(y)), "limiar_5pct": float(lim), "frac_classificados_simples": float(simples.mean()),
     "rede_todos": erros(rede, todos), "rede_classificados_simples": erros(rede, simples), "ajuste_1L1S_todos": erros(fit, todos)}
(ROOT / "models").mkdir(exist_ok=True)
json.dump(R, open(ROOT / "models/avaliacao_1l1s.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))

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
plt.savefig(ROOT / "figures/fig_1l1s.png", dpi=170); print("salvo: figures/fig_1l1s.png")
