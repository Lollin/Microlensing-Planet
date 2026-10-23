import sys, pathlib, argparse, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mlens.dataset_feat import build
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--engine", choices=["vbm", "pylima"], default="vbm"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--caustic-frac", type=float, default=None, help="fração ancorada na cáustica (padrão: config)"); ap.add_argument("--out", default=None); a = ap.parse_args()
    X, sc, y, d2 = build(a.n, a.seed, a.workers, a.engine, a.caustic_frac); out = a.out or f"data/processed/feat_{a.engine}.npz"
    np.savez_compressed(out, X=X, sc=sc, y=y, dchi2=d2); print(out, X.shape, "frac Δχ²>100:", float(np.mean(d2 > 100)))


if __name__ == "__main__":
    main()
