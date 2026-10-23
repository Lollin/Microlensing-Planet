import sys, pathlib, json, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sklearn.model_selection import train_test_split
from mlens import model as M
d = np.load("data/processed/train.npz"); X, y = d["X"], d["y"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
m = M.train(Xtr, ytr); yp = M.predict(m, Xte); r2 = M.evaluate(yte, yp)
M.save(m, "models/mlp_2l1s.joblib"); np.savez("data/processed/test_pred.npz", y=yte, yp=yp, Xte=Xte)
json.dump(r2, open("models/metrics.json", "w"), indent=2); json.dump({"n_train": len(Xtr), "n_test": len(Xte)}, open("models/meta.json", "w")); print(json.dumps(r2, indent=2))
