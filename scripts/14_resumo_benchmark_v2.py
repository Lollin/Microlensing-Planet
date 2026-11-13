"""Resumo estatístico do benchmark v2 (lê os JSON de 13_benchmark_pylima_v2.py).

Critérios de sucesso (todos contam as falhas como fracasso; nenhum evento é descartado):
  convergência  χ² ≤ χ²_ref + tol, com χ²_ref = menor χ² encontrado no evento por QUALQUER estratégia, incluindo o oráculo.
                Ou seja: a estratégia chegou ao melhor mínimo conhecido, e não só a "um χ² baixo".
  física        convergência E |Δ log q| < --tol-logq E | |log s| − |log s|_verd | < --tol-abslogs.
                O sinal de log s não é exigido (degenerescência s ↔ 1/s); o acerto do sinal é reportado à parte.
  χ²/N < 1,5    o critério antigo, mantido para comparação com a versão anterior do texto.
Estatística:
  - taxas com IC de 95% de Wilson;
  - teste de McNemar exato (pareado, mesmos eventos) entre pares de estratégias;
  - custo por sucesso = tempo total / número de sucessos; o FATOR DE ACELERAÇÃO é a razão desse custo entre a busca cega
    e a estratégia da rede, com IC de 95% por bootstrap sobre os eventos.
Uso:   python scripts/14_resumo_benchmark_v2.py
Saídas: models/benchmark_v2_resumo.json, figures/fig_benchmark_v2.png e linhas LaTeX no terminal."""
import sys, json, pathlib, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import binom
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(ROOT / "data/processed/benchmark_v2"))
ap.add_argument("--tol", type=float, default=10.0, help="tolerância em χ² para 'chegou ao melhor mínimo'")
ap.add_argument("--tol-logq", type=float, default=0.3); ap.add_argument("--tol-abslogs", type=float, default=0.1)
ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()

EV = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(pathlib.Path(a.dir).glob("evt_*.json"))]
if not EV: sys.exit(f"nenhum evento em {a.dir}")
SUF = pathlib.Path(a.dir).name.replace("benchmark_v2", "")          # ex.: "_rapido" no teste de fumaça
EST = ["oraculo", "rede_1ramo", "rede_2ramos", "cega_mesmo_orcamento", "cega_8"]
ROT = {"oraculo": "oráculo (partida verdadeira)", "rede_1ramo": "rede, um ramo", "rede_2ramos": "rede, dois ramos",
       "cega_mesmo_orcamento": "cega, mesmo orçamento", "cega_8": "cega, 8 partidas"}
n = len(EV); N = np.array([e["N"] for e in EV]); exc = np.array([e["excesso_chi2"] for e in EV])
vlq = np.array([e["verdadeiro"]["logq"] for e in EV]); vls = np.array([e["verdadeiro"]["logs"] for e in EV])
val = np.array([e["verdadeiro"]["alpha"] for e in EV])
g = lambda nome, k: np.array([e["estrategias"][nome].get(k, np.nan) for e in EV], float)
chi2 = {s: g(s, "chi2") for s in EST}; tempo = {s: g(s, "tempo_s") for s in EST}
ref = np.nanmin(np.vstack([chi2[s] for s in EST]), 0)
wa = lambda x, y: np.degrees(np.abs(np.angle(np.exp(1j * (x - y)))))
conv, fis, red, sinal = {}, {}, {}, {}
for s in EST:
    c = chi2[s]; ok = np.isfinite(c)
    conv[s] = ok & (c <= ref + a.tol)
    elq = np.abs(np.log10(g(s, "q")) - vlq); els = np.abs(np.abs(np.log10(g(s, "s"))) - np.abs(vls))
    fis[s] = conv[s] & (elq < a.tol_logq) & (els < a.tol_abslogs)
    red[s] = ok & (c / N < 1.5)
    sinal[s] = (np.sign(np.log10(g(s, "s"))) == np.sign(vls))


def wilson(k, m, z=1.96):
    if m == 0: return [float("nan")] * 2
    p = k / m; c = (p + z * z / (2 * m)) / (1 + z * z / m); h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / (1 + z * z / m)
    return [float(c - h), float(c + h)]


def taxa(v):
    k = int(v.sum()); return dict(k=k, n=len(v), frac=k / len(v), ic95=wilson(k, len(v)))


def mcnemar(x, y):
    """x, y booleanos pareados. b = só x acerta, c = só y acerta. p exato bilateral."""
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    p = 1.0 if b + c == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5)))
    return dict(so_primeira=b, so_segunda=c, p=p)


rng = np.random.default_rng(0)


def custo(t, ok):
    return float(np.nansum(t) / ok.sum()) if ok.sum() else float("inf")


def fator(s_lenta, s_rapida, crit):
    """Razão custo-por-sucesso(lenta) / custo-por-sucesso(rápida), com IC por bootstrap sobre os eventos."""
    f = lambda idx: custo(tempo[s_lenta][idx], crit[s_lenta][idx]) / custo(tempo[s_rapida][idx], crit[s_rapida][idx])
    v = f(np.arange(n)); bs = [f(rng.integers(0, n, n)) for _ in range(a.boot)]
    bs = np.array(bs)[np.isfinite(bs)]
    return dict(fator=v, ic95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else [np.nan] * 2)


R = {"n_eventos": n, "tol_chi2": a.tol, "tol_logq": a.tol_logq, "tol_abslogs": a.tol_abslogs, "estrategias": {}}
for s in EST:
    ok = np.isfinite(chi2[s])
    R["estrategias"][s] = dict(
        sem_resultado=int((~ok).sum()), falhas_de_ajuste=int(sum(e["estrategias"][s]["n_falhas"] for e in EV)),
        convergencia=taxa(conv[s]), fisica=taxa(fis[s]), chi2N_menor_1_5=taxa(red[s]),
        acerto_sinal_logs_entre_convergidos=taxa(sinal[s][conv[s]]) if conv[s].any() else None,
        tempo_mediano_s=float(np.nanmedian(tempo[s])), tempo_medio_s=float(np.nanmean(tempo[s])),
        custo_por_sucesso_conv_s=custo(tempo[s], conv[s]), custo_por_sucesso_fis_s=custo(tempo[s], fis[s]),
        n_ajustes_medio=float(np.mean([e["estrategias"][s]["n_ajustes"] for e in EV])),
        erro_mediano_logq=float(np.nanmedian(np.abs(np.log10(g(s, "q")) - vlq))),
        erro_mediano_abslogs=float(np.nanmedian(np.abs(np.abs(np.log10(g(s, "s"))) - np.abs(vls)))),
        erro_mediano_alpha_graus=float(np.nanmedian(wa(g(s, "alpha"), val))))
rq = np.array([e["rede"]["logq"] for e in EV]); rs = np.array([e["rede"]["abs_logs"] for e in EV]); ra = np.array([e["rede"]["alpha"] for e in EV])
R["rede_sem_ajuste"] = dict(tempo_mediano_s=float(np.median([e["rede"]["tempo_s"] for e in EV])),
                            erro_mediano_logq=float(np.median(np.abs(rq - vlq))), erro_mediano_abslogs=float(np.median(np.abs(rs - np.abs(vls)))),
                            erro_mediano_alpha_graus=float(np.median(wa(ra, val))))
PARES = [("rede_2ramos", "cega_mesmo_orcamento"), ("rede_2ramos", "cega_8"), ("rede_2ramos", "rede_1ramo")]
R["mcnemar"] = {f"{x} x {y}": {"convergencia": mcnemar(conv[x], conv[y]), "fisica": mcnemar(fis[x], fis[y])} for x, y in PARES}
R["fator_aceleracao"] = {f"cega_8 / {r}": {"convergencia": fator("cega_8", r, conv), "fisica": fator("cega_8", r, fis)}
                         for r in ("rede_2ramos", "rede_1ramo")}
R["fator_aceleracao"]["cega_mesmo_orcamento / rede_2ramos"] = {"convergencia": fator("cega_mesmo_orcamento", "rede_2ramos", conv),
                                                              "fisica": fator("cega_mesmo_orcamento", "rede_2ramos", fis)}
EDG = [-np.inf, 100, 1000, 1e5, np.inf]; LAB = ["<100", "100-1000", "1000-1e5", ">1e5"]
R["convergencia_por_excesso"] = {l: {"n": int(((exc > lo) & (exc <= hi)).sum()),
                                     **{s: float(conv[s][(exc > lo) & (exc <= hi)].mean()) if ((exc > lo) & (exc <= hi)).any() else None for s in EST}}
                                 for l, lo, hi in zip(LAB, EDG[:-1], EDG[1:])}
R["oraculo_chi2N_mediano"] = float(np.nanmedian(chi2["oraculo"] / N))
pathlib.Path(ROOT / "models").mkdir(exist_ok=True)
json.dump(R, open(ROOT / f"models/benchmark_v2_resumo{SUF}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)

# ---------------- terminal ----------------
pc = lambda t: f"{100 * t['frac']:3.0f}% [{100 * t['ic95'][0]:.0f}–{100 * t['ic95'][1]:.0f}]"
print(f"\nBenchmark v2: {n} eventos (nenhum descartado). Convergência: χ² ≤ χ²_ref + {a.tol:g}.  Física: + |Δlog q|<{a.tol_logq}, Δ|log s|<{a.tol_abslogs}")
print(f"{'estratégia':<30}{'converge':>18}{'física':>18}{'χ²/N<1,5':>18}{'tempo med.':>12}{'custo/sucesso':>15}{'sem result.':>12}")
for s in EST:
    r = R["estrategias"][s]
    print(f"{ROT[s]:<30}{pc(r['convergencia']):>18}{pc(r['fisica']):>18}{pc(r['chi2N_menor_1_5']):>18}"
          f"{r['tempo_mediano_s']:>10.1f} s{r['custo_por_sucesso_conv_s']:>13.1f} s{r['sem_resultado']:>12d}")
print("\nMcNemar (pareado; p exato):")
for k, v in R["mcnemar"].items():
    print(f"  {k:<40} convergência: só 1ª {v['convergencia']['so_primeira']}, só 2ª {v['convergencia']['so_segunda']}, p = {v['convergencia']['p']:.3f}"
          f"  | física: {v['fisica']['so_primeira']}/{v['fisica']['so_segunda']}, p = {v['fisica']['p']:.3f}")
print("\nFator de aceleração (custo por sucesso; IC 95% bootstrap):")
for k, v in R["fator_aceleracao"].items():
    print(f"  {k:<40} convergência: {v['convergencia']['fator']:.1f} [{v['convergencia']['ic95'][0]:.1f}; {v['convergencia']['ic95'][1]:.1f}]"
          f"   física: {v['fisica']['fator']:.1f} [{v['fisica']['ic95'][0]:.1f}; {v['fisica']['ic95'][1]:.1f}]")
print("\nConvergência por excesso de χ² do ajuste PSPL:")
for l, v in R["convergencia_por_excesso"].items():
    print(f"  {l:<10} n={v['n']:3d}  " + "  ".join(f"{s}={'—' if v[s] is None else f'{100 * v[s]:.0f}%'}" for s in EST))
print(f"\nχ²/N mediano do oráculo: {R['oraculo_chi2N_mediano']:.2f}  (se ≫ 1, o próprio modelo não descreve bem os dados: ex. ρ, mistura)")

print("\n=== Linhas LaTeX: estratégia & convergência & física & tempo mediano (s) & custo por sucesso (s) ===")
fmt = lambda x, d=1: f"{x:.{d}f}".replace(".", "{,}")
for s in EST:
    r = R["estrategias"][s]
    print(f"{ROT[s]} & {r['convergencia']['k']}/{n} & {r['fisica']['k']}/{n} & {fmt(r['tempo_mediano_s'])} & {fmt(r['custo_por_sucesso_conv_s'])} \\\\")

# ---------------- figura ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
x = np.arange(len(EST)); w = .38
for j, (crit, col, lab) in enumerate(((conv, "#3b6ea5", "convergência"), (fis, "#c0392b", "física"))):
    t = [taxa(crit[s]) for s in EST]; f = np.array([v["frac"] for v in t]); ic = np.array([v["ic95"] for v in t])
    ax[0].bar(x + (j - .5) * w, f, w, color=col, label=lab, yerr=[f - ic[:, 0], ic[:, 1] - f], capsize=3)
ax[0].set_xticks(x, [ROT[s].replace(", ", ",\n") for s in EST], fontsize=8); ax[0].set_ylim(0, 1.05)
ax[0].set_ylabel("fração de sucesso (IC 95%)"); ax[0].legend(); ax[0].set_title(f"(a) Sucesso em {n} eventos")
tm = [R["estrategias"][s]["tempo_mediano_s"] for s in EST]; cp = [R["estrategias"][s]["custo_por_sucesso_conv_s"] for s in EST]
ax[1].bar(x - w / 2, tm, w, color="#9bb7d4", label="tempo mediano por evento"); ax[1].bar(x + w / 2, cp, w, color="#3b6ea5", label="custo por sucesso")
ax[1].set_yscale("log"); ax[1].set_xticks(x, [ROT[s].replace(", ", ",\n") for s in EST], fontsize=8); ax[1].set_ylabel("s"); ax[1].legend(fontsize=8)
ax[1].set_title("(b) Tempo")
for s, c in zip(EST, ["k", "#e67e22", "#3b6ea5", "#7f7f7f", "#c0392b"]):
    d = np.sort(np.where(np.isfinite(chi2[s]), chi2[s] - ref, np.inf)); d = np.clip(d, 0.1, None)
    ax[2].step(d, np.arange(1, n + 1) / n, where="post", color=c, label=ROT[s])
ax[2].axvline(a.tol, color="gray", ls=":"); ax[2].set_xscale("log"); ax[2].set_xlabel(r"$\chi^2 - \chi^2_\mathrm{ref}$")
ax[2].set_ylabel("fração acumulada de eventos"); ax[2].legend(fontsize=7); ax[2].set_title(r"(c) Distância ao melhor mínimo conhecido")
plt.tight_layout(); (ROOT / "figures").mkdir(exist_ok=True)
plt.savefig(ROOT / f"figures/fig_benchmark_v2{SUF}.png", dpi=200); print(f"\nsalvo: models/benchmark_v2_resumo{SUF}.json e figures/fig_benchmark_v2{SUF}.png")
