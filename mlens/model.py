"""RNA (MLP) baseline em scikit-learn. Depois: trocar por CNN em PyTorch (mesmo formato X)."""
import numpy as np, joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

NAMES = ["t0", "u0", "tE", "logq", "logs", "cos_alpha", "sin_alpha"]


def flat(X):  # (N,2,B) -> (N,2B)
    return X.reshape(len(X), -1)


def train(Xtr, ytr, hidden=(256, 128, 64), seed=0):
    sx, sy = StandardScaler().fit(flat(Xtr)), StandardScaler().fit(ytr)
    net = MLPRegressor(hidden_layer_sizes=hidden, early_stopping=True, validation_fraction=0.1,
                       n_iter_no_change=15, max_iter=400, random_state=seed)
    net.fit(sx.transform(flat(Xtr)), sy.transform(ytr))
    return {"net": net, "sx": sx, "sy": sy}


def predict(m, X):
    return m["sy"].inverse_transform(m["net"].predict(m["sx"].transform(flat(X))))


def evaluate(y, yp):
    r2 = {n: float(r2_score(y[:, i], yp[:, i])) for i, n in enumerate(NAMES)}
    a, ap = np.arctan2(y[:, 6], y[:, 5]), np.arctan2(yp[:, 6], yp[:, 5])
    d = np.angle(np.exp(1j * (ap - a)))                       # erro angular em (-pi, pi]
    r2["alpha_erro_medio_deg"] = float(np.degrees(np.mean(np.abs(d))))
    return r2


save = lambda m, path: joblib.dump(m, path)
load = lambda path: joblib.load(path)
