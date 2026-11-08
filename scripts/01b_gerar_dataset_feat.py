import sys, pathlib, argparse, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mlens.dataset_feat import build
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--engine", choices=["vbm", "pylima"], default="vbm"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--caustic-frac", type=float, default=None, help="fração ancorada na cáustica (padrão: config)"); ap.add_argument("--out", default=None)
    ap.add_argument("--p-null", type=float, default=0.0, help="fração de eventos SEM planeta (lente simples)")
    ap.add_argument("--null-vbm", action="store_true", help="controle: eventos sem planeta calculados pelo VBMicrolensing (q=1e-9, s=10) em vez da fórmula analítica"); a = ap.parse_args()
    X, sc, y, d2, pl = build(a.n, a.seed, a.workers, a.engine, a.caustic_frac, a.p_null, a.null_vbm); out = a.out or f"data/processed/feat_{a.engine}.npz"
    np.savez_compressed(out, X=X, sc=sc, y=y, dchi2=d2, planet=pl)
    print(out, X.shape, "frac Δχ²>100:", float(np.mean(d2 > 100)), "| fração com planeta:", float(pl.mean()))


if __name__ == "__main__":
    main()
