"""Análise de detecção: probabilidade de planeta da rede (v3) x critério clássico do excesso de χ² da lente simples.
Não retreina nada. Usa:
  data/processed/test_pred_uniform_<run>.npz   (eventos COM planeta: p_planet e excesso de χ²)
  data/processed/test_null_pplanet_<run>.npy   (eventos SEM planeta: p_planet)
  data/processed/test_null.npz                 (eventos SEM planeta: excesso de χ²)
Uso:   python scripts/10_deteccao.py --run v3
Saídas: models/deteccao_<run>.json e figures/fig_deteccao_<run>.png

Ideia: escolhe-se o limiar de cada detector que dá uma taxa fixa de falsos positivos (5% e 1%) nos eventos SEM planeta;
com esse limiar, mede-se a completeza (fração de planetas detectados) em cada faixa de excesso de χ²."""
import sys, json, pathlib, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--run", default="v3")
ap.add_argument("--null-file", default=None, help="outro conjunto sem planeta (ex.: test_null_vbm.npz); exige --model")
ap.add_argument("--model", default=None, help="modelo para calcular p_planeta no --null-file"); a = ap.parse_args()
P = np.load(ROOT / f"data/processed/test_pred_uniform_{a.run}.npz")
pp_pl, d2_pl = P["p_planet"], P["d2"]
if a.null_file:                                                     # controle: calcula p_planeta agora, com o modelo dado
    sys.path.insert(0, str(ROOT)); from mlens.torch_cnn import predict
    z = np.load(a.null_file); pp_nu = predict(a.model, z["X"], z["sc"])["p_planet"]; d2_nu = z["dchi2"]
    a.run = a.run + "_" + pathlib.Path(a.null_file).stem
else:
    pp_nu = np.load(ROOT / f"data/processed/test_null_pplanet_{a.run}.npy")
    d2_nu = np.load(ROOT / "data/processed/test_null.npz")["dchi2"]

EDG = [-np.inf, 10, 100, 1000, np.inf]; LAB = ["<10", "10-100", "100-1000", ">1000"]
lab = np.r_[np.ones(len(pp_pl)), np.zeros(len(pp_nu))]
R = {"n_com_planeta": int(len(pp_pl)), "n_sem_planeta": int(len(pp_nu)),
     "AUC_rede": float(roc_auc_score(lab, np.r_[pp_pl, pp_nu])),
     "AUC_excesso_chi2": float(roc_auc_score(lab, np.r_[d2_pl, d2_nu])), "por_taxa_falso_positivo": {}}
for fpr in (0.05, 0.01):
    linha = {}
    for nome, sp, sn in (("rede", pp_pl, pp_nu), ("excesso_chi2", d2_pl, d2_nu)):
        lim = float(np.quantile(sn, 1 - fpr))                      # limiar que deixa passar só fpr dos eventos sem planeta
        comp = {l: float(np.mean(sp[(d2_pl > lo) & (d2_pl <= hi)] > lim)) for l, lo, hi in zip(LAB, EDG[:-1], EDG[1:])}
        linha[nome] = dict(limiar=lim, completeza_total=float(np.mean(sp > lim)), completeza_por_excesso=comp)
    R["por_taxa_falso_positivo"][f"{int(fpr * 100)}%"] = linha
print(json.dumps(R, indent=1, ensure_ascii=False))
(ROOT / "models").mkdir(exist_ok=True)
json.dump(R, open(ROOT / f"models/deteccao_{a.run}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
b = np.linspace(0, 1, 41)
ax[0].hist(pp_nu, b, alpha=.6, density=True, color="#7f7f7f", label="sem planeta")
ax[0].hist(pp_pl[d2_pl <= 10], b, alpha=.6, density=True, color="#9bb7d4", label=r"com planeta, excesso $\leq 10$")
ax[0].hist(pp_pl[d2_pl > 100], b, alpha=.6, density=True, color="#c0392b", label=r"com planeta, excesso $> 100$")
ax[0].set_xlabel("probabilidade de planeta dada pela rede"); ax[0].set_ylabel("densidade"); ax[0].legend(fontsize=9)
ax[0].set_title("(a) Saída da rede")
x = np.arange(4); w = .38; f5 = R["por_taxa_falso_positivo"]["5%"]
ax[1].bar(x - w / 2, [f5["rede"]["completeza_por_excesso"][l] for l in LAB], w, color="#3b6ea5", label="rede")
ax[1].bar(x + w / 2, [f5["excesso_chi2"]["completeza_por_excesso"][l] for l in LAB], w, color="#c0392b", label=r"excesso de $\chi^2$")
ax[1].set_xticks(x, LAB); ax[1].set_ylim(0, 1.05); ax[1].legend()
ax[1].set_xlabel(r"excesso de $\chi^2$ do ajuste 1L1S"); ax[1].set_ylabel("completeza (fração detectada)")
ax[1].set_title("(b) Completeza com 5% de falsos positivos")
plt.tight_layout(); (ROOT / "figures").mkdir(exist_ok=True)
plt.savefig(ROOT / f"figures/fig_deteccao_{a.run}.png", dpi=200); print(f"salvo: figures/fig_deteccao_{a.run}.png")
