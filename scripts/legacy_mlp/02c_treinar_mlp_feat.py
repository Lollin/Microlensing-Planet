"""Baseline sklearn com features de resíduo; avalia por subconjunto de Δχ² (detectabilidade da anomalia)."""
import sys, pathlib, json, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
d = np.load("data/processed/feat_vbm.npz"); X, sc, y, d2 = d["X"], d["sc"], d["y"], d["dchi2"]
F = np.concatenate([X.reshape(len(X), -1), sc], 1); n = int(.8 * len(y))
sx, sy = StandardScaler().fit(F[:n]), StandardScaler().fit(y[:n])
net = MLPRegressor(hidden_layer_sizes=(256, 128, 64), early_stopping=True, n_iter_no_change=12, max_iter=300, random_state=0).fit(sx.transform(F[:n]), sy.transform(y[:n]))
yp = sy.inverse_transform(net.predict(sx.transform(F[n:]))); yt, dd = y[n:], d2[n:]
names = ["t0", "u0", "tE", "logq", "logs", "cos_alpha", "sin_alpha"]; out = {}
for lab, m in [("todos", dd > -1e18), ("dchi2>100", dd > 100), ("dchi2>1000", dd > 1000)]:
    r = {k: float(r2_score(yt[m, i], yp[m, i])) for i, k in enumerate(names)}
    a, ap = np.arctan2(yt[m, 6], yt[m, 5]), np.arctan2(yp[m, 6], yp[m, 5])
    r["abs_logs"] = float(r2_score(np.abs(yt[m, 4]), np.abs(yp[m, 4])))
    r["alpha_erro_deg"] = float(np.degrees(np.abs(np.angle(np.exp(1j * (ap - a))))).mean()); r["n"] = int(m.sum()); out[lab] = r
    print(lab, {k: round(v, 2) for k, v in r.items()})
json.dump(out, open("models/metrics_feat_mlp.json", "w"), indent=2)
