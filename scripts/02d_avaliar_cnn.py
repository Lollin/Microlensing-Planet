"""Avalia a CNN já treinada em OUTRO conjunto (ex.: simulado com pyLIMA) — teste de generalização entre simuladores.
Uso:  python scripts/02d_avaliar_cnn.py --data data/processed/feat_pylima.npz --model models/cnn_2l1s.pt"""
import sys, pathlib, json, argparse, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sklearn.metrics import r2_score
from mlens.torch_cnn import predict

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/processed/feat_pylima.npz"); ap.add_argument("--model", default="models/cnn_2l1s.pt")
ap.add_argument("--min-dchi2", type=float, default=100.0); ap.add_argument("--out", default="models/metrics_cnn_pylima.json"); a = ap.parse_args()
d = np.load(a.data); m = d["dchi2"] > a.min_dchi2; X, sc, y = d["X"][m], d["sc"][m], d["y"][m]
P = predict(a.model, X, sc); at = np.arctan2(y[:, 6], y[:, 5]) % (2 * np.pi)
r = dict(n=int(m.sum()), t0=r2_score(y[:, 0], P["t0"]), u0=r2_score(y[:, 1], P["u0"]), tE=r2_score(y[:, 2], P["tE"]), logq=r2_score(y[:, 3], P["logq"]),
         abs_logs=r2_score(np.abs(y[:, 4]), P["abs_logs"]), alpha_erro_deg=float(np.degrees(np.abs(np.angle(np.exp(1j * (P["alpha"] - at))))).mean()),
         acc_s_gt1=float(np.mean((P["p_wide"] > .5) == (y[:, 4] > 0))))
print({k: round(float(v), 3) for k, v in r.items()}); json.dump(r, open(a.out, "w"), indent=2)
