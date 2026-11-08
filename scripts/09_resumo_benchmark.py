"""Resumo do benchmark CNN x pyLIMA (lê a saída de 04_benchmark_pylima.py).
Uso:   python scripts/09_resumo_benchmark.py                 (padrão: data/processed/benchmark.csv)
Saídas: figures/fig_benchmark.png e models/benchmark_resumo.json"""
import sys, json, pathlib, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--csv", default=str(ROOT / "data/processed/benchmark.csv"))
ap.add_argument("--bom", type=float, default=1.5, help="χ²/N abaixo disso = ajuste bem-sucedido"); a = ap.parse_args()
d = pd.read_csv(a.csv)
wa = lambda x, y: np.degrees(np.abs(np.angle(np.exp(1j * (np.asarray(x) - np.asarray(y))))))
ok = d.fitNN_chi2N.notna() & d.blind_chi2N.notna(); o = d[ok]
med = lambda x: float(np.median(x))
R = dict(n=int(len(d)), n_ajustados=int(ok.sum()), n_falha_ambos=int((d.fitNN_chi2N.isna() & d.blind_chi2N.isna()).sum()),
         tempo_rede_s=med(d.t_nn_s), tempo_ajuste_rede_s=med(o.fitNN_s), tempo_busca_cega_s=med(o.blind_s),
         razao_tempo=med(o.blind_s / o.fitNN_s),
         sucesso_rede=int((o.fitNN_chi2N < a.bom).sum()), sucesso_cega=int((o.blind_chi2N < a.bom).sum()),
         rede_igual_ou_melhor=int((o.fitNN_chi2N <= 1.05 * o.blind_chi2N).sum()))
for k in ("logq", "logs"):
    R[f"erro_{k}"] = dict(rede=med(abs(d["nn_" + k] - d["true_" + k])), ajuste_rede=med(abs(o["fitNN_" + k] - o["true_" + k])),
                          busca_cega=med(abs(o["blind_" + k] - o["true_" + k])))
R["erro_alpha_graus"] = dict(rede=med(wa(d.nn_alpha, d.true_alpha)), ajuste_rede=med(wa(o.fitNN_alpha, o.true_alpha)),
                             busca_cega=med(wa(o.blind_alpha, o.true_alpha)))
if "br_chi2N" in d:                                              # versão 3: ajuste em dois ramos
    ob = d[ok & d.br_chi2N.notna()]
    R["dois_ramos"] = dict(n=int(len(ob)), tempo_s=med(ob.br_s), sucesso=int((ob.br_chi2N < a.bom).sum()),
                           igual_ou_melhor_que_cega=int((ob.br_chi2N <= 1.05 * ob.blind_chi2N).sum()),
                           erro_logq=med(abs(ob.br_logq - ob.true_logq)), erro_logs=med(abs(ob.br_logs - ob.true_logs)),
                           erro_alpha_graus=med(wa(ob.br_alpha, ob.true_alpha)))
    g = ob[ob.br_chi2N < a.bom]                                  # topologia certa, entre os ajustes bem-sucedidos
    R["dois_ramos"]["topologia_certa_entre_sucessos"] = f"{int((g.br_topologia == g.true_topologia).sum())} de {len(g)}"
if "nn_p_planet" in d and d.nn_p_planet.notna().any():
    R["p_planeta_mediana_eventos_com_planeta"] = med(d.nn_p_planet)
print(json.dumps(R, indent=1, ensure_ascii=False))
json.dump(R, open(ROOT / "models/benchmark_resumo.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
lo, hi = 0.5, max(o.fitNN_chi2N.max(), o.blind_chi2N.max()) * 2
ax[0].loglog(o.blind_chi2N, o.fitNN_chi2N, "o", ms=8, color="#c0392b", label="um ramo (sinal da rede)")
if "br_chi2N" in d:
    ax[0].loglog(o.blind_chi2N, o.br_chi2N, "s", ms=7, mfc="none", color="#3b6ea5", label="dois ramos (close/wide)")
ax[0].legend(fontsize=9, loc="upper left")
ax[0].plot([lo, hi], [lo, hi], "k--", lw=1); ax[0].axhline(a.bom, color="gray", ls=":"); ax[0].axvline(a.bom, color="gray", ls=":")
ax[0].set_xlabel(r"$\chi^2/N$ — busca com várias partidas"); ax[0].set_ylabel(r"$\chi^2/N$ — ajuste iniciado pela rede")
ax[0].set_title("(a) Qualidade do ajuste (abaixo da diagonal: rede melhor)")
lab = ["rede\n(inferência)", "ajuste iniciado\npela rede", "busca com\nvárias partidas"]
v = [R["tempo_rede_s"], R["tempo_ajuste_rede_s"], R["tempo_busca_cega_s"]]
if "dois_ramos" in R:
    lab.insert(2, "dois ramos\n(close/wide)"); v.insert(2, R["dois_ramos"]["tempo_s"])
x = np.arange(len(v))
ax[1].bar(x, v, color=["#3b6ea5", "#9bb7d4", "#6c8ebf", "#c0392b"][:len(v)]); ax[1].set_yscale("log")
ax[1].set_xticks(x, lab)
for xi, vi in zip(x, v): ax[1].text(xi, vi * 1.3, f"{vi:.3g} s", ha="center")
ax[1].set_ylim(top=max(v) * 6)
ax[1].set_ylabel("tempo mediano por evento (s)"); ax[1].set_title("(b) Tempo de processamento")
plt.tight_layout(); (ROOT / "figures").mkdir(exist_ok=True)
plt.savefig(ROOT / "figures/fig_benchmark.png", dpi=200); print("salvo: figures/fig_benchmark.png")
