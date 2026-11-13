"""Métricas das Tabelas 3 e 4 restritas ao regime planetário (tópico 3 da revisão da banca).

A priori de log10 q vai de −4 a −1. Para uma estrela de 0,3 M☉, q = 0,1 corresponde a ~31 massas de Júpiter (anã marrom).
Este script repete as métricas só para q < q_max (padrão 0,03, ~9 M_J para 0,3 M☉; convenção comum para "planeta" em
microlente) e, para comparação, para q ≥ q_max. Não retreina nada.
Uso:   python scripts/15_regime_planetario.py --run final            (ou --qmax 0.01, etc.)
Saída: models/regime_planetario_<run>.json"""
import sys, json, pathlib, argparse, numpy as np
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mlens import metrics as M

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="final"); ap.add_argument("--qmax", type=float, default=0.03); a = ap.parse_args()
d = np.load(ROOT / f"data/processed/test_pred_uniform_{a.run}.npz")
y, ex = d["y"], d["d2"]; P = {k: d[k] for k in ("t0", "u0", "tE", "logq", "abs_logs", "logs", "alpha", "p_wide")}
if "planet" in d.files:
    k = d["planet"] > 0; y, ex = y[k], ex[k]; P = {kk: v[k] for kk, v in P.items()}
lim = np.log10(a.qmax); R = {"qmax": a.qmax}
for nome, m in ((f"q < {a.qmax:g}", y[:, 3] < lim), (f"q >= {a.qmax:g}", y[:, 3] >= lim)):
    Pm = {kk: v[m] for kk, v in P.items()}
    R[nome] = M.full_report(y[m], Pm, ex[m]); M.print_report(R[nome], f"{nome}  (n = {int(m.sum())}, fração {m.mean():.0%})")
    for g in ("all", "exc>100"):
        s = R[nome][g]
        if s.get("n", 0) >= 20:
            print(f"  R² [{g}]: " + "  ".join(f"{k_} {v_:.2f}" for k_, v_ in s["r2"].items()))
json.dump(R, open(ROOT / f"models/regime_planetario_{a.run}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\nsalvo: models/regime_planetario_{a.run}.json")
