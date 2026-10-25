"""Figuras do banner: topologias de cáustica, evento exemplo (cáustica + dados + resíduo) e previsto×verdadeiro da CNN."""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mlens import plots, simulate
from mlens.features import featurize

plots.fig_topologies("figures/fig_topologias.png")
# evento exemplo: planeta com travessia de cáustica, observado por 3 sítios (mude os parâmetros à vontade)
V = plots._vbm(); s0, q0 = 1.3, 5e-3; u0, al = plots.through_caustic(V, s0, q0, "right", 0.5)
p = {"t0": 100.0, "u0": u0, "tE": 30.0, "logq": float(np.log10(q0)), "logs": float(np.log10(s0)), "alpha": al}
rng = np.random.default_rng(11); p, data = simulate.simulate_event(rng, p)
X, sc, d2, fit = featurize(data); plots.fig_event(p, data, fit, "figures/fig_evento.png"); print("Δχ² do evento exemplo:", round(d2))
import os
tag = "uniform" if os.path.exists("data/processed/test_pred_uniform.npz") else "mix"
t = np.load(f"data/processed/test_pred_{tag}.npz"); P = {k: t[k] for k in ["t0", "u0", "tE", "logq", "abs_logs", "logs", "alpha", "p_wide"]}
plots.fig_resultados(t["y"], P, t["d2"], "figures/fig_resultados_cnn.png"); print("figuras ok; população de teste:", tag)
