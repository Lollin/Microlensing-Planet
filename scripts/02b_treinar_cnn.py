"""Treina a CNN 1D e avalia com MÉTRICAS DE ERRO.
Ex.:  python scripts/02b_treinar_cnn.py --train-data data/processed/train_mix.npz --test-data data/processed/test_uniform.npz --epochs 30
Divisão do arquivo de treino: 70% treino, 10% validação (escolhe a melhor época), 20% teste 'mix'.
O teste 'uniform' (priori uniforme, sem ancoragem em cáusticas) é um arquivo separado, gerado com outra semente.
Versão 2:  --uchan (canal u(τ)), --target x (alvo |s − 1/s|), --wbin (peso de q, s, α na perda), --run NOME (não sobrescreve a versão 1).
Versão 3:  --null (treino com eventos sem planeta: 9ª saída = probabilidade de planeta; q, s, α mascarados na perda)
           --null-test ARQ (conjunto só de lentes simples, para medir falsos positivos).
Ex.:  python scripts/02b_treinar_cnn.py --uchan --target x --null --train-data data/processed/train_null.npz --run v3"""
import sys, os, pathlib, json, argparse, copy, numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mlens.torch_cnn import Net, prep_x, make_targets, decode
from mlens import metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--train-data", default="data/processed/train_mix.npz"); ap.add_argument("--test-data", default="data/processed/test_uniform.npz")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--min-excess", type=float, default=0.0, help="se > 0, treina só com eventos de excesso de chi2 do 1L1S acima disso (padrão: todos)")
ap.add_argument("--n-train", type=int, default=0, help="usa só N eventos de treino (curva de aprendizado)")
ap.add_argument("--uchan", action="store_true", help="acrescenta o canal log10 u(τ)")
ap.add_argument("--target", choices=["abslogs", "x"], default="abslogs", help="alvo da separação: |log s| ou |s − 1/s|")
ap.add_argument("--wbin", type=float, default=1.0, help="peso na perda de log q, separação e α (t0, u0, tE têm peso 0,5)")
ap.add_argument("--null", action="store_true", help="9ª saída: probabilidade de planeta (requer 'planet' no arquivo de treino)")
ap.add_argument("--null-test", default="data/processed/test_null.npz", help="teste só com lentes simples (falsos positivos)")
ap.add_argument("--run", default="", help="nome do experimento: salva models/cnn_<run>.pt, models/metrics_cnn_<run>.json etc.")
ap.add_argument("--out", default="models/cnn_2l1s.pt"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
d = np.load(a.train_data); X, sc, y, d2 = d["X"], d["sc"], d["y"], d["dchi2"]; n = len(y)
PL = d["planet"].astype(np.float32) if "planet" in d.files else np.ones(n, np.float32)
if a.null and PL.min() == 1: sys.exit("ERRO: --null exige um arquivo de treino com eventos sem planeta (gere com --p-null).")
idx = np.random.RandomState(a.seed).permutation(n); tr, va, te = idx[:int(.7 * n)], idx[int(.7 * n):int(.8 * n)], idx[int(.8 * n):]
if a.min_excess > 0:
    tr = tr[d2[tr] > a.min_excess]
if a.n_train:
    tr = tr[:a.n_train]
sfx = f"_{a.run}" if a.run else ""
if a.run and a.out == "models/cnn_2l1s.pt": a.out = f"models/cnn_{a.run}.pt"
reg, flag = make_targets(y, a.target); trp = tr[PL[tr] > 0]; ym, ys = reg[trp].mean(0), reg[trp].std(0) + 1e-8; scm, scs = sc[tr].mean(0), sc[tr].std(0) + 1e-8
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", dev, "| treino:", len(tr), "validação:", len(va), "teste-mix:", len(te))
T = lambda i: (torch.tensor(prep_x(X[i], sc[i], a.uchan)).to(dev), torch.tensor((sc[i] - scm) / scs, dtype=torch.float32).to(dev),
               torch.tensor((reg[i] - ym) / ys).to(dev), torch.tensor(flag[i]).to(dev), torch.tensor(PL[i]).to(dev))
Xtr, Str, Ytr, Ftr, Ptr = T(tr); Xva, Sva, Yva, Fva, Pva = T(va)
net = Net(cin=3 + int(a.uchan), nsc=sc.shape[1], nout=9 if a.null else 8).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((len(tr) + a.bs - 1) // a.bs))
w = torch.tensor([.5, .5, .5, a.wbin, a.wbin, a.wbin, a.wbin], device=dev); bce = nn.BCEWithLogitsLoss()
bce_n = nn.BCEWithLogitsLoss(reduction="none")


def lossf(o, Y, F, Pl):
    """PSPL (t0, u0, tE) em todos os eventos; q, separação, α e sinal só nos eventos com planeta; + BCE da probabilidade de planeta."""
    se = w * (o[:, :7] - Y) ** 2; m = Pl[:, None]
    reg = (se[:, :3].sum(1) + (se[:, 3:] * m).sum(1)).sum() / (len(Y) * 7)
    sgn = (bce_n(o[:, 7], F) * Pl).sum() / Pl.sum().clamp(min=1)
    return reg + 0.3 * sgn + (0.3 * bce(o[:, 8], Pl) if a.null else 0.0)
best = (1e9, None, -1)
for ep in range(a.epochs):
    net.train(); perm = torch.randperm(len(tr), device=dev); tot = 0
    for i in range(0, len(tr), a.bs):
        b = perm[i:i + a.bs]; loss = lossf(net(Xtr[b], Str[b]), Ytr[b], Ftr[b], Ptr[b])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item() * len(b)
    net.eval()
    with torch.no_grad():
        lv = lossf(net(Xva, Sva), Yva, Fva, Pva).item()
    if lv < best[0]:
        best = (lv, copy.deepcopy(net.state_dict()), ep)
    if ep % 5 == 0 or ep == a.epochs - 1:
        print(f"ep {ep:3d}  perda treino {tot / len(tr):.4f}  validação {lv:.4f}")
net.load_state_dict(best[1]); net.eval(); print("melhor época (validação):", best[2])


def predict_raw(Xe, Sce):
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xe), 1000):
            outs.append(net(torch.tensor(prep_x(Xe[i:i + 1000], Sce[i:i + 1000], a.uchan)).to(dev), torch.tensor((Sce[i:i + 1000] - scm) / scs, dtype=torch.float32).to(dev)).cpu().numpy())
    return np.concatenate(outs)


def run(Xe, Sce, ye, d2e, tag, pl=None):
    P = decode(predict_raw(Xe, Sce), ym, ys, a.target)
    extra = {} if pl is None else {"planet": pl}
    np.savez(f"data/processed/test_pred_{tag}{sfx}.npz", y=ye, d2=d2e, **extra, **{k: np.asarray(v) for k, v in P.items()})
    if pl is not None and pl.min() == 0:                      # avalia regressão só nos eventos com planeta
        k = pl > 0; P = {kk: np.asarray(v)[k] for kk, v in P.items()}; ye, d2e = ye[k], d2e[k]
    rep = M.full_report(ye, P, d2e); M.print_report(rep, f"teste {tag}")
    if "p_planet" in P:
        rep["deteccao_por_excesso"] = {lab: float(np.mean(P["p_planet"][(d2e > lo) & (d2e <= hi)] > 0.5))
                                       for lab, lo, hi in zip(M.EXC_LABELS, M.EXC_EDGES[:-1], M.EXC_EDGES[1:]) if ((d2e > lo) & (d2e <= hi)).sum()}
        print("  fração com p_planeta > 0,5 por faixa de excesso:", {k_: round(v_, 2) for k_, v_ in rep["deteccao_por_excesso"].items()})
    return rep


res = {"mix": run(X[te], sc[te], y[te], d2[te], "mix", PL[te])}
meta = {"cfg": dict(uchan=a.uchan, target=a.target, wbin=a.wbin, null=a.null, train_data=a.train_data), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test_mix": int(len(te)), "best_epoch": int(best[2]), "epochs": a.epochs}
if os.path.exists(a.test_data):
    u = np.load(a.test_data); res["uniform"] = run(u["X"], u["sc"], u["y"], u["dchi2"], "uniform"); meta["n_test_uniform"] = int(len(u["y"]))
else:
    print("AVISO: --test-data não encontrado; avaliando só o teste 'mix' (priori enviesada para cáusticas).")
if a.null and os.path.exists(a.null_test):
    z = np.load(a.null_test); pp = 1 / (1 + np.exp(-predict_raw(z["X"], z["sc"])[:, 8]))
    res["falso_positivo"] = dict(n=int(len(pp)), frac_p_planeta_maior_05=float(np.mean(pp > 0.5)), p_planeta_mediana=float(np.median(pp)))
    np.save(f"data/processed/test_null_pplanet{sfx}.npy", pp)
    print(f"== teste sem planeta (n={len(pp)}): falsos positivos (p_planeta > 0,5) = {100 * np.mean(pp > 0.5):.1f}%")
json.dump(res, open(f"models/metrics_cnn{sfx}.json", "w"), indent=2); json.dump(meta, open(f"models/meta_cnn{sfx}.json", "w"))
torch.save(dict(state=net.state_dict(), ym=ym, ys=ys, scm=scm, scs=scs, cfg=dict(uchan=a.uchan, target=a.target, null=a.null)), a.out); print("salvo em", a.out)
