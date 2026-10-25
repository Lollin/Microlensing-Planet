"""Analise de s SEM retreinar. Usa data/processed/test_pred_uniform.npz.
Uso: python scripts/06_analise_s.py
Saidas: figures/fig_s_analise.png, figures/fig_erro_vs_excesso.png, models/analise_s.json"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

d = np.load("data/processed/test_pred_uniform.npz")
y, p, ex = d["y"], d["p_wide"], d["d2"]
ls = y[:, 4]; lab = ls > 0; pred = p > 0.5
RES = 0.15                      # regiao ressonante: |log s| < 0.15  (0.71 < s < 1.41)
nr = np.abs(ls) >= RES
EDG = [-np.inf, 10, 100, 1000, np.inf]; LAB = ["<10", "10-100", "100-1000", ">1000"]
acc = lambda m: float((pred[m] == lab[m]).mean())
auc = lambda m: float(roc_auc_score(lab[m], p[m]))
out = {"n": int(len(ls)), "frac_resonante": float(1 - nr.mean()),
       "acc_todos": acc(np.ones_like(lab)), "acc_fora_res": acc(nr), "auc_fora_res": auc(nr)}
print(out)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
# (a) acerto do sinal por faixa de |log s|
b = np.array([0, .05, .1, .15, .2, .25, .302]); c = (b[1:] + b[:-1]) / 2
a1 = [acc((np.abs(ls) >= lo) & (np.abs(ls) < hi)) for lo, hi in zip(b[:-1], b[1:])]
ax[0].bar(c, a1, width=.045, color="#3b6ea5"); ax[0].axhline(.5, color="k", ls="--", lw=1)
ax[0].axvline(RES, color="gray", ls=":"); ax[0].set_ylim(0, 1)
ax[0].set_xlabel(r"$|\log_{10} s|$ verdadeiro"); ax[0].set_ylabel("acerto do sinal"); ax[0].set_title("(a) Acerto close/wide vs |log s|")
# (b) acerto do sinal por excesso, todos vs fora da regiao ressonante
x = np.arange(4); w = .38; A, B = [], []
for lo, hi in zip(EDG[:-1], EDG[1:]):
    m = (ex > lo) & (ex <= hi); A.append(acc(m)); B.append(acc(m & nr))
ax[1].bar(x - w/2, A, w, label="todos", color="#9bb7d4"); ax[1].bar(x + w/2, B, w, label=r"$|\log s|\geq0{,}15$", color="#3b6ea5")
ax[1].axhline(.5, color="k", ls="--", lw=1); ax[1].set_xticks(x, LAB); ax[1].set_ylim(0, 1); ax[1].legend()
ax[1].set_xlabel(r"excesso de $\chi^2$ do ajuste 1L1S"); ax[1].set_title("(b) Acerto do sinal vs anomalia")
# (c) cobertura x acerto (classificacao seletiva)
conf = np.abs(p - .5) * 2; o = np.argsort(-conf)
for nm, m, col in [("todos", np.ones_like(lab), "#9bb7d4"), (r"excesso$>100$", ex > 100, "#3b6ea5")]:
    idx = o[m[o]]; cov = np.arange(1, len(idx) + 1) / len(idx)
    cum = np.cumsum(pred[idx] == lab[idx]) / np.arange(1, len(idx) + 1)
    ax[2].plot(cov[20:], cum[20:], label=nm, color=col)
ax[2].axhline(.5, color="k", ls="--", lw=1); ax[2].set_xlabel("fração de eventos mantidos (mais confiantes primeiro)")
ax[2].set_ylabel("acerto do sinal"); ax[2].set_ylim(.4, 1); ax[2].legend(); ax[2].set_title("(c) Classificação seletiva")
plt.tight_layout(); plt.savefig("figures/fig_s_analise.png", dpi=200); plt.close()

# Figura 3 ampliada: erro vs excesso (q, alpha, sinal)
def werr(a, b): return np.abs((a - b + 180) % 360 - 180)
alt = np.degrees(np.arctan2(y[:, 6], y[:, 5])) % 360; alp = np.degrees(d["alpha"]) if d["alpha"].max() < 7 else d["alpha"]
eq = np.abs(d["logq"] - y[:, 3]); ea = werr(alp, alt)
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
def line(a, v, ylab, ttl, ref=None):
    md, lo, hi = [], [], []
    for l, h in zip(EDG[:-1], EDG[1:]):
        m = (ex > l) & (ex <= h); md.append(np.median(v[m])); lo.append(np.percentile(v[m], 16)); hi.append(np.percentile(v[m], 84))
    md, lo, hi = map(np.array, (md, lo, hi))
    a.errorbar(x, md, yerr=[md - lo, hi - md], fmt="o-", color="#c0392b", lw=2.5, ms=9, capsize=5)
    if ref is not None: a.axhline(ref, color="k", ls="--", lw=1.2, label="sem informação (mediana da priori)"); a.legend()
    a.set_xticks(x, LAB); a.set_xlabel(r"excesso de $\chi^2$ do ajuste 1L1S"); a.set_ylabel(ylab); a.set_title(ttl, fontsize=13); a.set_ylim(bottom=0)
    return md
mq = line(ax[0], eq, r"erro absoluto em $\log_{10} q$ (dex)", r"(a) Razão de massa $q$")
ma = line(ax[1], ea, r"erro angular em $\alpha$ (graus)", r"(b) Ângulo $\alpha$", ref=90)
ax[2].bar(x, B, .55, color="#c0392b"); ax[2].axhline(.5, color="k", ls="--", lw=1.2); ax[2].set_ylim(0, 1)
ax[2].set_xticks(x, LAB); ax[2].set_xlabel(r"excesso de $\chi^2$ do ajuste 1L1S"); ax[2].set_ylabel("acerto close/wide"); ax[2].set_title(r"(c) Sinal de $\log s$ ($|\log s|\geq0{,}15$)", fontsize=13)
plt.tight_layout(); plt.savefig("figures/fig_erro_vs_excesso.png", dpi=220); plt.close()
out.update(erro_logq_mediana_por_bin=dict(zip(LAB, map(float, mq))), erro_alpha_mediana_por_bin=dict(zip(LAB, map(float, ma))),
           acc_sinal_por_bin=dict(zip(LAB, A)), acc_sinal_fora_res_por_bin=dict(zip(LAB, B)))
json.dump(out, open("models/analise_s.json", "w"), indent=1); print(json.dumps(out, indent=1))
