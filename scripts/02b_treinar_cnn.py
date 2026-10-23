"""Treina a CNN 1D. Ex.:  python scripts/02b_treinar_cnn.py --data data/processed/feat_vbm.npz --epochs 60 --min-dchi2 100"""
import sys, pathlib, json, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sklearn.metrics import r2_score
from mlens.torch_cnn import Net, prep_x, make_targets, decode

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/processed/feat_vbm.npz")
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--min-dchi2", type=float, default=0.0, help="treina só com eventos de anomalia detectável"); ap.add_argument("--out", default="models/cnn_2l1s.pt")
a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0)
d = np.load(a.data); X, sc, y, d2 = d["X"], d["sc"], d["y"], d["dchi2"]
n = int(.8 * len(y)); tr = np.arange(n); te = np.arange(n, len(y)); tr = tr[d2[tr] > a.min_dchi2]
reg, flag = make_targets(y); ym, ys = reg[tr].mean(0), reg[tr].std(0) + 1e-8; scm, scs = sc[tr].mean(0), sc[tr].std(0) + 1e-8
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", dev, "| treino:", len(tr), "teste:", len(te))
T = lambda i: (torch.tensor(prep_x(X[i])), torch.tensor((sc[i] - scm) / scs, dtype=torch.float32),
               torch.tensor((reg[i] - ym) / ys), torch.tensor(flag[i]))
Xtr, Str, Ytr, Ftr = [t.to(dev) for t in T(tr)]; Xte, Ste, _, _ = [t.to(dev) for t in T(te)]
net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((len(tr) + a.bs - 1) // a.bs))
w = torch.tensor([.5, .5, .5, 1, 1, 1, 1], device=dev); bce = nn.BCEWithLogitsLoss()
for ep in range(a.epochs):
    net.train(); perm = torch.randperm(len(tr), device=dev); tot = 0
    for i in range(0, len(tr), a.bs):
        b = perm[i:i + a.bs]; o = net(Xtr[b], Str[b])
        loss = (w * (o[:, :7] - Ytr[b]) ** 2).mean() + 0.3 * bce(o[:, 7], Ftr[b])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item() * len(b)
    if ep % 5 == 0 or ep == a.epochs - 1: print(f"ep {ep:3d}  loss {tot / len(tr):.4f}")
net.eval()
with torch.no_grad(): out = net(Xte, Ste).cpu().numpy()
P = decode(out, ym, ys); yt = y[te]; dd = d2[te]; res = {}
for lab, m in [("todos", dd > -1e18), ("dchi2>100", dd > 100), ("dchi2>1000", dd > 1000)]:
    at = np.arctan2(yt[m, 6], yt[m, 5]) % (2 * np.pi); e = np.degrees(np.abs(np.angle(np.exp(1j * (P["alpha"][m] - at))))).mean()
    res[lab] = dict(n=int(m.sum()), t0=r2_score(yt[m, 0], P["t0"][m]), u0=r2_score(yt[m, 1], P["u0"][m]), tE=r2_score(yt[m, 2], P["tE"][m]),
                    logq=r2_score(yt[m, 3], P["logq"][m]), abs_logs=r2_score(np.abs(yt[m, 4]), P["abs_logs"][m]),
                    alpha_erro_deg=float(e), acc_s_gt1=float(np.mean((P["p_wide"][m] > .5) == (yt[m, 4] > 0))))
    print(lab, {k: round(float(v), 3) for k, v in res[lab].items()})
json.dump(res, open("models/metrics_cnn.json", "w"), indent=2); json.dump({"n_train": int(len(tr)), "n_test": int(len(te))}, open("models/meta_cnn.json", "w"))
np.savez("data/processed/test_pred_cnn.npz", y=yt, d2=dd, **{k: np.asarray(v) for k, v in P.items()})
torch.save(dict(state=net.state_dict(), ym=ym, ys=ys, scm=scm, scs=scs), a.out); print("salvo em", a.out)
