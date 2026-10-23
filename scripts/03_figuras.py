import sys, pathlib, json, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mlens import simulate, plots, model as M
rng = np.random.default_rng(7)
p0 = {"t0": 100.0, "u0": 0.12, "tE": 30.0, "logq": -3.0, "logs": float(np.log10(1.1)), "alpha": 0.9}
p, data = simulate.simulate_event(rng, p0); plots.fig_example(p, data, "figures/fig1_exemplo_2L1S.png")
t = np.load("data/processed/test_pred.npz"); r2 = json.load(open("models/metrics.json"))
plots.fig_pred_true(t["y"], t["yp"], r2, "figures/fig2_previsto_vs_verdadeiro.png")
r2_1l1s = {"t0": 0.975, "u0": 0.945, "tE": 0.987}   # valores do resumo submetido
plots.fig_r2_bars(r2_1l1s, {k: r2[k] for k in ["t0", "u0", "tE", "logq", "logs"]}, "figures/fig3_R2_1L1S_vs_2L1S.png")
plots.fig_loss(M.load("models/mlp_2l1s.joblib"), "figures/fig4_curva_perda.png"); print("ok")

plots.fig_geometry("figures/fig0_geometria.png")
