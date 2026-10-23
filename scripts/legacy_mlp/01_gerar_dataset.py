import sys, pathlib, argparse, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mlens.dataset import build
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="data/processed/train.npz")
ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
X, y = build(a.n, a.seed, a.workers); np.savez_compressed(a.out, X=X, y=y); print(a.out, X.shape, y.shape)
