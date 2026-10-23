"""Benchmark: RNA/CNN (+ ajuste pyLIMA inicializado por ela) vs. busca 'cega' multi-partida do pyLIMA, em eventos simulados com pyLIMA.
Ex.:  python scripts/04_benchmark_pylima.py --n 20 --thin 2 --model models/cnn_2l1s.pt   (~3-5 min/evento por causa do ajuste cego)"""
import sys, pathlib, argparse, time, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mlens.simulate_pylima import simulate_event_pylima, JD0
from mlens.features import featurize
from mlens.pylima_fit import fit_usbl, blind_multistart
from mlens.torch_cnn import predict

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--thin", type=int, default=2)
ap.add_argument("--model", default="models/cnn_2l1s.pt"); ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--out", default="data/processed/benchmark.csv")
a = ap.parse_args(); rows = []
for i in range(a.n):
    rng = np.random.default_rng(a.seed + i); p, data, _ = simulate_event_pylima(rng)
    X, sc, d2, fit = featurize(data); t = time.time(); P = predict(a.model, X[None], sc[None]); t_nn = time.time() - t
    thin = {k: (v[0][::a.thin] + JD0, v[1][::a.thin], v[2][::a.thin]) for k, v in data.items()}; N = sum(len(v[0]) for v in thin.values())
    base = [JD0 + fit["t0"], fit["u0"], fit["tE"]]
    s_nn = 10 ** float(P["logs"][0]); q_nn = 10 ** float(P["logq"][0]); al_nn = float(P["alpha"][0])
    try: r_nn = fit_usbl(thin, [*base, s_nn, q_nn, al_nn])
    except Exception as e: r_nn = None
    r_bl = blind_multistart(thin, base)
    f = lambda r, k: np.nan if r is None else r[k]
    rows.append(dict(i=i, dchi2=d2, true_logq=p["logq"], true_logs=p["logs"], true_alpha=p["alpha"],
        nn_logq=float(P["logq"][0]), nn_logs=float(P["logs"][0]), nn_alpha=al_nn, t_nn_s=t_nn,
        fitNN_chi2N=f(r_nn, "chi2") / N, fitNN_logq=np.log10(f(r_nn, "q")), fitNN_logs=np.log10(f(r_nn, "s")), fitNN_alpha=f(r_nn, "alpha"), fitNN_s=f(r_nn, "seconds"),
        blind_chi2N=f(r_bl, "chi2") / N, blind_logq=np.log10(f(r_bl, "q")), blind_logs=np.log10(f(r_bl, "s")), blind_alpha=f(r_bl, "alpha"), blind_s=f(r_bl, "seconds")))
    pd.DataFrame(rows).to_csv(a.out, index=False); print(f"[{i + 1}/{a.n}] Δχ²={d2:.0f}  χ²/N: NN-init={rows[-1]['fitNN_chi2N']:.2f}  cego={rows[-1]['blind_chi2N']:.2f}  tempo: {t_nn:.3f}s vs {rows[-1]['blind_s']:.0f}s")
print("salvo em", a.out)
