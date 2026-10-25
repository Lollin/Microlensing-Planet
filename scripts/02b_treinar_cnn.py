"""Treina a CNN 1D e avalia com MÉTRICAS DE ERRO.
Ex.:  python scripts/02b_treinar_cnn.py --train-data data/processed/train_mix.npz --test-data data/processed/test_uniform.npz --epochs 30
Divisão do arquivo de treino: 70% treino, 10% validação (escolhe a melhor época), 20% teste 'mix'.
O teste 'uniform' (priori uniforme, sem ancoragem em cáusticas) é um arquivo separado, gerado com outra semente."""
import sys, os, pathlib, json, argparse, copy, numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mlens.torch_cnn import Net, prep_x, make_targets, decode
from mlens import metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--train-data", default="data/processed/train_mix.npz"); ap.add_argument("--test-data", default="data/processed/test_uniform.npz")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--min-excess", type=float, default=0.0, help="se > 0, treina só com eventos de excesso de chi2 do 1L1S acima disso (padrão: todos)")
ap.add_argument("--n-train", type=int, default=0, help="usa só N eventos de treino (curva de aprendizado)")
ap.add_argument("--out", default="models/cnn_2l1s.pt"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
d = np.load(a.train_data); X, sc, y, d2 = d["X"], d["sc"], d["y"], d["dchi2"]; n = len(y)
idx = np.random.RandomState(a.seed).permutation(n); tr, va, te = idx[:int(.7 * n)], idx[int(.7 * n):int(.8 * n)], idx[int(.8 * n):]
if a.min_excess > 0:
    tr = tr[d2[tr] > a.min_excess]
if a.n_train:
    tr = tr[:a.n_train]
reg, flag = make_targets(y); ym, ys = reg[tr].mean(0), reg[tr].std(0) + 1e-8; scm, scs = sc[tr].mean(0), sc[tr].std(0) + 1e-8
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", dev, "| treino:", len(tr), "validação:", len(va), "teste-mix:", len(te))
T = lambda i: (torch.tensor(prep_x(X[i])).to(dev), torch.tensor((sc[i] - scm) / scs, dtype=torch.float32).to(dev),
               torch.tensor((reg[i] - ym) / ys).to(dev), torch.tensor(flag[i]).to(dev))
Xtr, Str, Ytr, Ftr = T(tr); Xva, Sva, Yva, Fva = T(va)
net = Net(nsc=sc.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((len(tr) + a.bs - 1) // a.bs))
w = torch.tensor([.5, .5, .5, 1, 1, 1, 1], device=dev); bce = nn.BCEWithLogitsLoss()
lossf = lambda o, Y, F: (w * (o[:, :7] - Y) ** 2).mean() + 0.3 * bce(o[:, 7], F)
best = (1e9, None, -1)
for ep in range(a.epochs):
    net.train(); perm = torch.randperm(len(tr), device=dev); tot = 0
    for i in range(0, len(tr), a.bs):
        b = perm[i:i + a.bs]; loss = lossf(net(Xtr[b], Str[b]), Ytr[b], Ftr[b])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item() * len(b)
    net.eval()
    with torch.no_grad():
        lv = lossf(net(Xva, Sva), Yva, Fva).item()
    if lv < best[0]:
        best = (lv, copy.deepcopy(net.state_dict()), ep)
    if ep % 5 == 0 or ep == a.epochs - 1:
        print(f"ep {ep:3d}  perda treino {tot / len(tr):.4f}  validação {lv:.4f}")
net.load_state_dict(best[1]); net.eval(); print("melhor época (validação):", best[2])


def run(Xe, Sce, ye, d2e, tag):
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xe), 1000):
            outs.append(net(torch.tensor(prep_x(Xe[i:i + 1000])).to(dev), torch.tensor((Sce[i:i + 1000] - scm) / scs, dtype=torch.float32).to(dev)).cpu().numpy())
    P = decode(np.concatenate(outs), ym, ys)
    np.savez(f"data/processed/test_pred_{tag}.npz", y=ye, d2=d2e, **{k: np.asarray(v) for k, v in P.items()})
    rep = M.full_report(ye, P, d2e); M.print_report(rep, f"teste {tag}"); return rep


res = {"mix": run(X[te], sc[te], y[te], d2[te], "mix")}
meta = {"n_train": int(len(tr)), "n_val": int(len(va)), "n_test_mix": int(len(te)), "best_epoch": int(best[2]), "epochs": a.epochs}
if os.path.exists(a.test_data):
    u = np.load(a.test_data); res["uniform"] = run(u["X"], u["sc"], u["y"], u["dchi2"], "uniform"); meta["n_test_uniform"] = int(len(u["y"]))
else:
    print("AVISO: --test-data não encontrado; avaliando só o teste 'mix' (priori enviesada para cáusticas).")
json.dump(res, open("models/metrics_cnn.json", "w"), indent=2); json.dump(meta, open("models/meta_cnn.json", "w"))
torch.save(dict(state=net.state_dict(), ym=ym, ys=ys, scm=scm, scs=scs), a.out); print("salvo em", a.out)
