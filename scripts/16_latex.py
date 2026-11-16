"""Gera, a partir dos arquivos JSON de resultados, tudo o que a monografia usa: nenhum número é digitado à mão.

Saídas (pasta --out, padrão latex_tcc/), com a mesma estrutura do projeto no Overleaf:
  resultados/numeros.tex     um \\newcommand por número citado no texto (ex.: \\rTz = erro mediano da rede em t0)
  resultados/tab_*.tex       o corpo (tabular) de cada tabela de resultados
  figuras/*.jpg              as figuras de resultados, já com os nomes usados no texto
Para atualizar a monografia depois de uma nova rodada: rode este script e envie as pastas resultados/ e figuras/ ao Overleaf
(substituindo as existentes). Um número ainda não disponível aparece em vermelho no PDF, como [pendente: nome].

Uso:   python scripts/16_latex.py                      (lê models/, figures/ e data/processed/)
       python scripts/16_latex.py --models PASTA --figures PASTA --out PASTA"""
import sys, json, math, pathlib, argparse
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--models", default=str(ROOT / "models")); ap.add_argument("--figures", default=str(ROOT / "figures"))
ap.add_argument("--data", default=str(ROOT / "data/processed")); ap.add_argument("--out", default=str(ROOT / "latex_tcc"))
a = ap.parse_args()
MOD, FIG, DAT, OUT = map(pathlib.Path, (a.models, a.figures, a.data, a.out))
(OUT / "resultados").mkdir(parents=True, exist_ok=True); (OUT / "figuras").mkdir(parents=True, exist_ok=True)
FALTAM = []


def J(nome):
    f = MOD / nome
    if not f.exists():
        FALTAM.append(nome); return None
    return json.load(open(f, encoding="utf-8"))


def g(d, *ks):
    """d[k1][k2]... ou None se algo faltar."""
    for k in ks:
        if d is None: return None
        try: d = d[k]
        except (KeyError, IndexError, TypeError): return None
    return d


def ok(x):
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def br(x, n=2):
    """Número com vírgula decimal, pronto para texto ou modo matemático."""
    if not ok(x): return None
    s = f"{x:.{n}f}"
    if float(s) == 0: s = s.lstrip("-")
    return (r"\ensuremath{-}" if s.startswith("-") else "") + s.lstrip("-").replace(".", "{,}")


def mil(x):
    return None if not ok(x) else f"{int(round(x)):,}".replace(",", r"\,")


def pc(x, n=0):
    return None if not ok(x) else br(100 * x, n)


def sig(x, n=2):
    """n algarismos significativos."""
    if not ok(x): return None
    if x == 0: return "0"
    d = max(n - 1 - int(math.floor(math.log10(abs(x)))), 0)
    return br(x, d)


def sci(x, n=1):
    if not ok(x) or x <= 0: return None
    e = int(math.floor(math.log10(x))); m = x / 10 ** e
    if round(m, n) >= 10: m, e = m / 10, e + 1
    return f"{br(m, n)} \\times 10^{{{e}}}"


def ic(v, f=br, n=2, sep="; "):
    return None if v is None or not all(ok(x) for x in v) else f"[{f(v[0], n)}{sep}{f(v[1], n)}]"


def pval(p):
    if not ok(p): return None
    return r"$p < 0{,}001$" if p < 0.001 else f"$p = {br(p, 3 if p < 0.01 else 2)}$"


MAC = {}


def M(nome, valor):
    assert nome.isalpha(), nome
    assert nome not in MAC, "macro repetida: " + nome
    MAC[nome] = valor


P = lambda v: v if v is not None else None                         # só para legibilidade

# ------------------------------------------------------------------------------------------------------------------
# Leitura
# ------------------------------------------------------------------------------------------------------------------
SEM = ["final", "final_s1", "final_s2"]; ABL = ["abl_sem_u", "abl_sem_u_s1", "abl_sem_u_s2"]
MET = {r: g(J(f"metrics_cnn_{r}.json"), "uniform") for r in SEM + ABL + ["abl_u", "final_det"]}
R2 = {r: J(f"r2_{r}.json") for r in SEM + ABL + ["abl_u"]}
RVP = {r: J(f"rede_vs_pspl_{r}.json") for r in SEM + ["abl_sem_u", "final_det"]}
ANS, PLA = J("analise_s_final.json"), J("regime_planetario_final.json")
DET, DETP, DETC = J("deteccao_final_det.json"), J("deteccao_final_det_logq-1.52.json"), J("deteccao_final_det_test_null_vbm.json")
L1, BEN = J("avaliacao_1l1s_final_det.json"), J("benchmark_v2_resumo.json")
OG = {e: J(f"ogle_{e}.json") for e in ("2012-BLG-0406", "2016-BLG-1195")}
MAN, META = J("MANIFESTO_final.json"), J("meta_cnn_final.json")
m, mf = g(MET, "final", "all"), g(MET, "final", "exc>100")
r2k = lambda run, grp: next((R2[run][k] for k in (R2[run] or {}) if (k == "todos") == (grp == "todos")), None)
rv = lambda grp, par, campo, run="final": g(RVP, run, "grupos", grp, par, campo)

# referência sem informação: erro mediano de prever sempre a mediana (calculada no próprio conjunto de teste, se disponível)
REF = dict(t0=19.68, u0=0.254, tE=30.2, logq=0.76, logs=0.15, abs_logs=None)
try:
    import numpy as np
    y = np.load(DAT / "test_uniform.npz")["y"]; md = lambda v: float(np.median(np.abs(v - np.median(v))))
    REF = dict(t0=md(y[:, 0]), u0=md(y[:, 1]), tE=float(100 * np.median(np.abs(np.median(y[:, 2]) / y[:, 2] - 1))), logq=md(y[:, 3]),
               logs=md(y[:, 4]), abs_logs=md(np.abs(y[:, 4])))
except Exception:
    pass

# ------------------------------------------------------------------------------------------------------------------
# Macros: conjunto de teste e desempenho global
# ------------------------------------------------------------------------------------------------------------------
nb = lambda b: g(MET, "final", "bins", b, "n")
M("nTeste", mil(g(m, "n"))); M("nForte", mil(g(mf, "n")))
M("pctForte", pc(g(mf, "n") / g(m, "n")) if ok(g(m, "n")) and ok(g(mf, "n")) else None)
M("pctFraco", pc(nb("<10") / g(m, "n")) if ok(nb("<10")) else None)
M("nMuitoForte", mil(nb(">1000")))
for nome, par, n in (("Tz", "t0", 2), ("Uz", "u0", 3), ("TePct", "tE_pct", 1), ("Lq", "logq", 2), ("Ls", "logs", 2), ("Als", "abs_logs", 3)):
    M("r" + nome, br(g(m, par, "med_abs"), n)); M("r" + nome + "Forte", br(g(mf, par, "med_abs"), n))
M("rAl", br(g(m, "alpha_deg", "med_abs"), 0)); M("rAlForte", br(g(mf, "alpha_deg", "med_abs"), 0))
M("rAlTrinta", pc(g(m, "alpha_deg", "frac_lt30"))); M("rAlTrintaForte", pc(g(mf, "alpha_deg", "frac_lt30")))
M("sAcc", pc(g(m, "sign", "acc"))); M("sAccForte", pc(g(mf, "sign", "acc"))); M("sAuc", br(g(m, "sign", "auc"))); M("sAucForte", br(g(mf, "sign", "auc")))
M("sAccAlto", pc(g(ANS, "acc_sinal_por_bin", ">1000"))); M("sAccAltoNaoRes", pc(g(ANS, "acc_sinal_fora_res_por_bin", ">1000")))
M("sAccBaixo", pc(g(ANS, "acc_sinal_por_bin", "<10")))
M("sTopVinte", pc(g(ANS, "acc_20pct_mais_confiantes", "todos"))); M("sTopVinteForte", pc(g(ANS, "acc_20pct_mais_confiantes", "excesso>100")))
M("refTz", br(REF["t0"], 1)); M("refUz", br(REF["u0"], 2)); M("refTePct", br(REF["tE"], 0)); M("refLq", br(REF["logq"], 2))
for nome, k in (("Tz", "t0"), ("Uz", "u0"), ("TeLog", "log tE"), ("TeLin", "tE (linear)"), ("Lq", "log q"), ("Als", "|log s|")):
    M("rd" + nome, br(g(r2k("final", "todos"), k, "r2"))); M("rd" + nome + "Forte", br(g(r2k("final", "forte"), k, "r2")))
# por faixa de excesso
for suf, b in (("Baixo", "<10"), ("Medio", "100-1000"), ("Alto", ">1000")):
    bb = g(MET, "final", "bins", b)
    M("bLq" + suf, br(g(bb, "logq", "med_abs"), 2)); M("bAl" + suf, br(g(bb, "alpha_deg", "med_abs"), 0))
    M("bTz" + suf, br(g(bb, "t0", "med_abs"), 2)); M("bUz" + suf, br(g(bb, "u0", "med_abs"), 3)); M("bTePct" + suf, br(g(bb, "tE_pct", "med_abs"), 1))

# ------------------------------------------------------------------------------------------------------------------
# Macros: rede x ajuste PSPL
# ------------------------------------------------------------------------------------------------------------------
dex2pct = lambda x: None if not ok(x) else 100 * (10 ** x - 1)
for suf, grp in (("", "todos"), ("Forte", "excesso>100"), ("Baixo", "excesso <10"), ("Alto", "excesso >1000")):
    for nome, par, n in (("Tz", "t0", 3), ("Uz", "u0", 3)):
        M("p" + nome + suf, br(rv(grp, par, "erro_mediano_pspl"), n)); M("v" + nome + suf, br(rv(grp, par, "erro_mediano_rede"), n))
    M("pTePct" + suf, br(dex2pct(rv(grp, "tE", "erro_mediano_pspl")), 1)); M("vTePct" + suf, br(dex2pct(rv(grp, "tE", "erro_mediano_rede")), 1))
    for nome, par in (("Tz", "t0"), ("Uz", "u0"), ("Te", "tE")):
        M("melhor" + nome + suf, pc(rv(grp, par, "frac_rede_melhor"))); M("beta" + nome + suf, br(rv(grp, par, "beta")))
        M("beta" + nome + suf + "IC", ic(rv(grp, par, "ic95_beta")))
        rz = rv(grp, par, "razao_rede_pspl"); M("fator" + nome + suf, br(1 / rz, 1) if ok(rz) and rz > 0 else None)
        M("corr" + nome + suf, br(rv(grp, par, "corr")))
M("pWilcoxTzBaixo", pval(rv("excesso <10", "t0", "p_wilcoxon")))
bs = [rv("todos", p_, "beta", r) for r in SEM for p_ in ("u0", "tE") if ok(rv("todos", p_, "beta", r))]
M("betaMinSementes", br(min(bs)) if bs else None); M("betaMaxSementes", br(max(bs)) if bs else None)
bt = [rv("todos", "t0", "beta", r) for r in SEM if ok(rv("todos", "t0", "beta", r))]
M("betaTzMinSementes", br(min(bt)) if bt else None); M("betaTzMaxSementes", br(max(bt)) if bt else None)
M("nSementes", str(sum(MET[r] is not None for r in SEM)))
LIM = g(RVP, "final", "pspl_limitado")
M("nLim", mil(g(LIM, "n"))); M("pctLim", pc(g(LIM, "frac"), 1)); M("pctLimAlto", pc(g(LIM, "frac_no_excesso>1000")))
M("pctLimSaoAlto", pc(g(LIM, "frac_dos_limitados_com_excesso>1000")))
M("limUzPspl", br(g(LIM, "u0", "erro_mediano_pspl"), 2)); M("limUzRede", br(g(LIM, "u0", "erro_mediano_rede"), 2))
M("limTePspl", br(g(LIM, "tE", "erro_mediano_pspl"), 2)); M("limTeRede", br(g(LIM, "tE", "erro_mediano_rede"), 2))
M("limTzPspl", br(g(LIM, "t0", "erro_mediano_pspl"), 2)); M("limTzRede", br(g(LIM, "t0", "erro_mediano_rede"), 2))

# ------------------------------------------------------------------------------------------------------------------
# Macros: regime planetário, detecção, 1L1S
# ------------------------------------------------------------------------------------------------------------------
kp = next((k for k in (PLA or {}) if k.startswith("q <")), None); kn = next((k for k in (PLA or {}) if k.startswith("q >=")), None)
M("plN", mil(g(PLA, kp, "all", "n"))); M("plPct", pc(g(PLA, kp, "all", "n") / g(m, "n")) if ok(g(PLA, kp, "all", "n")) and ok(g(m, "n")) else None)
M("plLq", br(g(PLA, kp, "all", "logq", "med_abs"))); M("plLqForte", br(g(PLA, kp, "exc>100", "logq", "med_abs")))
M("plRdLq", br(g(PLA, kp, "all", "r2", "logq"))); M("plRdLqForte", br(g(PLA, kp, "exc>100", "r2", "logq")))
M("plAlTrintaForte", pc(g(PLA, kp, "exc>100", "alpha_deg", "frac_lt30"))); M("plSAccForte", pc(g(PLA, kp, "exc>100", "sign", "acc")))
M("plTz", br(g(PLA, kp, "all", "t0", "med_abs"), 3)); M("plUz", br(g(PLA, kp, "all", "u0", "med_abs"), 3)); M("plTePct", br(g(PLA, kp, "all", "tE_pct", "med_abs"), 1))
M("npTz", br(g(PLA, kn, "all", "t0", "med_abs"), 2)); M("npUz", br(g(PLA, kn, "all", "u0", "med_abs"), 3)); M("npTePct", br(g(PLA, kn, "all", "tE_pct", "med_abs"), 1))
d5 = lambda D_, det, *ks: g(D_, "por_taxa_falso_positivo", "5%", det, *ks)
for suf, D_ in (("", DET), ("Pl", DETP)):
    M("dAucRede" + suf, br(g(D_, "AUC_rede"))); M("dAucChi" + suf, br(g(D_, "AUC_excesso_chi2")))
    M("dCompRede" + suf, pc(d5(D_, "rede", "completeza_total"))); M("dCompChi" + suf, pc(d5(D_, "excesso_chi2", "completeza_total")))
    for b, bn in (("<10", "Baixo"), ("10-100", "MedBaixo"), ("100-1000", "Medio"), (">1000", "Alto")):
        M("dRede" + bn + suf, pc(d5(D_, "rede", "completeza_por_excesso", b))); M("dChi" + bn + suf, pc(d5(D_, "excesso_chi2", "completeza_por_excesso", b)))
M("dNpl", mil(g(DET, "n_com_planeta"))); M("dNnull", mil(g(DET, "n_sem_planeta"))); M("dNplPl", mil(g(DETP, "n_com_planeta")))
M("dLimChi", br(d5(DET, "excesso_chi2", "limiar"), 0)); M("dLimRede", br(d5(DET, "rede", "limiar"), 2))
M("dCompRedeUm", pc(g(DET, "por_taxa_falso_positivo", "1%", "rede", "completeza_total")))
M("dCompChiUm", pc(g(DET, "por_taxa_falso_positivo", "1%", "excesso_chi2", "completeza_total")))
M("dAucCtrl", br(g(DETC, "AUC_rede"), 3)); M("dAucRedeTres", br(g(DET, "AUC_rede"), 3))
M("dDesvioNull", br(g(DET, "sem_planeta", "desvio_padrao_excesso_chi2"), 0)); M("dMedNull", br(g(DET, "sem_planeta", "mediana_p_planeta"), 2))
md_ = g(MET, "final_det", "all")
M("detUz", br(g(md_, "u0", "med_abs"), 3)); M("detLq", br(g(md_, "logq", "med_abs"), 2)); M("detTePct", br(g(md_, "tE_pct", "med_abs"), 1))
M("lN", mil(g(L1, "n"))); M("lFrac", pc(g(L1, "frac_classificados_simples"), 1)); M("lFracIC", ic(g(L1, "ic95_frac_classificados_simples"), pc, 1, "; "))
for nome, k, n in (("Tz", "t0_med_dias", 3), ("Uz", "u0_med", 3), ("TePct", "tE_med_pct", 1)):
    M("lRede" + nome, br(g(L1, "rede_todos", k), n)); M("lFit" + nome, br(g(L1, "ajuste_1L1S_todos", k), n))
M("lRedeRdUz", br(g(L1, "rede_todos", "R2", "u0"), 3)); M("lFitRdUz", br(g(L1, "ajuste_1L1S_todos", "R2", "u0")))

# ------------------------------------------------------------------------------------------------------------------
# Macros: benchmark com o pyLIMA
# ------------------------------------------------------------------------------------------------------------------
EST = [("Or", "oraculo"), ("Um", "rede_1ramo"), ("Dois", "rede_2ramos"), ("Orc", "cega_mesmo_orcamento"), ("Oito", "cega_8")]
be = lambda s, *ks: g(BEN, "estrategias", s, *ks)
M("bN", mil(g(BEN, "n_eventos"))); M("bTolChi", br(g(BEN, "tol_chi2"), 0)); M("bTolLq", br(g(BEN, "tol_logq"), 1)); M("bTolLs", br(g(BEN, "tol_abslogs"), 1))
for suf, s in EST:
    M("bConv" + suf, str(be(s, "convergencia", "k")) if ok(be(s, "convergencia", "k")) else None); M("bFis" + suf, str(be(s, "fisica", "k")) if ok(be(s, "fisica", "k")) else None)
    M("bConvPct" + suf, pc(be(s, "convergencia", "frac"))); M("bFisPct" + suf, pc(be(s, "fisica", "frac")))
    M("bConvIC" + suf, ic(be(s, "convergencia", "ic95"), pc, 0, "--")); M("bRed" + suf, str(be(s, "chi2N_menor_1_5", "k")) if ok(be(s, "chi2N_menor_1_5", "k")) else None)
    M("bTempo" + suf, sig(be(s, "tempo_mediano_s"), 2)); M("bCusto" + suf, br(be(s, "custo_por_sucesso_conv_s"), 0))
    M("bErroLq" + suf, br(be(s, "erro_mediano_logq"), 2)); M("bAjustes" + suf, br(be(s, "n_ajustes_medio"), 1))
M("bTempoRede", br(g(BEN, "rede_sem_ajuste", "tempo_mediano_s"), 2)); M("bRedeLq", br(g(BEN, "rede_sem_ajuste", "erro_mediano_logq"), 2))
M("bRedeAls", br(g(BEN, "rede_sem_ajuste", "erro_mediano_abslogs"), 2)); M("bRedeAl", br(g(BEN, "rede_sem_ajuste", "erro_mediano_alpha_graus"), 0))
mc = lambda par, crit, campo: g(BEN, "mcnemar", par, crit, campo)
for suf, par in (("Orc", "rede_2ramos x cega_mesmo_orcamento"), ("Oito", "rede_2ramos x cega_8"), ("Um", "rede_2ramos x rede_1ramo")):
    for c, cn in (("convergencia", "Conv"), ("fisica", "Fis")):
        M("mc" + cn + suf, pval(mc(par, c, "p"))); M("mc" + cn + suf + "So", str(mc(par, c, "so_primeira")) if ok(mc(par, c, "so_primeira")) else None)
        M("mc" + cn + suf + "Outra", str(mc(par, c, "so_segunda")) if ok(mc(par, c, "so_segunda")) else None)
fa = lambda par, crit, campo: g(BEN, "fator_aceleracao", par, crit, campo)
for suf, par in (("OitoDois", "cega_8 / rede_2ramos"), ("OitoUm", "cega_8 / rede_1ramo"), ("OrcDois", "cega_mesmo_orcamento / rede_2ramos")):
    M("ac" + suf, br(fa(par, "convergencia", "fator"), 1)); M("ac" + suf + "IC", ic(fa(par, "convergencia", "ic95"), br, 1))
    M("acFis" + suf, br(fa(par, "fisica", "fator"), 1)); M("acFis" + suf + "IC", ic(fa(par, "fisica", "ic95"), br, 1))
cx = lambda b, s: g(BEN, "convergencia_por_excesso", b, s)
M("bxNAlto", str(cx(">1e5", "n")) if ok(cx(">1e5", "n")) else None); M("bxNBaixo", str(cx("<100", "n")) if ok(cx("<100", "n")) else None)
for suf, s in EST:
    M("bxAlto" + suf, pc(cx(">1e5", s))); M("bxBaixo" + suf, pc(cx("<100", s)))
M("bChiOr", br(g(BEN, "oraculo_chi2N_mediano"), 2))
M("bSinalDois", pc(be("rede_2ramos", "acerto_sinal_logs_entre_convergidos", "frac"))); M("bSinalOito", pc(be("cega_8", "acerto_sinal_logs_entre_convergidos", "frac")))

# ------------------------------------------------------------------------------------------------------------------
# Macros: eventos reais do OGLE
# ------------------------------------------------------------------------------------------------------------------
for suf, e in (("A", "2012-BLG-0406"), ("B", "2016-BLG-1195")):
    o = OG[e]
    M("o" + suf + "RedeQ", sci(g(o, "rede", "q"))); M("o" + suf + "RedeSw", br(g(o, "rede", "s_wide"))); M("o" + suf + "RedeSc", br(g(o, "rede", "s_close")))
    M("o" + suf + "RedePw", br(g(o, "rede", "p_wide"))); M("o" + suf + "PsplUz", sig(g(o, "pspl", "u0"), 2)); M("o" + suf + "PsplTe", sig(g(o, "pspl", "tE"), 2))
    M("o" + suf + "FitQ", sci(g(o, "ajuste", "q"))); M("o" + suf + "FitS", br(g(o, "ajuste", "s"))); M("o" + suf + "FitUz", sig(g(o, "ajuste", "u0"), 2))
    M("o" + suf + "FitTe", sig(g(o, "ajuste", "tE"), 2)); M("o" + suf + "FitRho", br(g(o, "ajuste", "rho"), 3)); M("o" + suf + "ChiN", br(g(o, "ajuste", "chi2N")))
    M("o" + suf + "Dchi", mil(g(o, "ajuste", "dchi2_pspl")) if ok(g(o, "ajuste", "dchi2_pspl")) and g(o, "ajuste", "dchi2_pspl") >= 100 else br(g(o, "ajuste", "dchi2_pspl"), 0))
    M("o" + suf + "DchiRamos", br(g(o, "ajuste", "dchi2_outro_ramo"), 0)); M("o" + suf + "Erros", br(g(o, "fator_erros"), 2)); M("o" + suf + "N", mil(g(o, "N_janela")))

# confirmação (semente nova, avaliada uma vez)
CONF = g(J("confirmacao_final.json"), "uniform"); CRV = J("rede_vs_pspl_confirmacao_final.json")
cm, cmf = g(CONF, "all"), g(CONF, "exc>100")
M("cN", mil(g(cm, "n"))); M("cTz", br(g(cm, "t0", "med_abs"), 3)); M("cUz", br(g(cm, "u0", "med_abs"), 3)); M("cTePct", br(g(cm, "tE_pct", "med_abs"), 1))
M("cLq", br(g(cm, "logq", "med_abs"))); M("cLqForte", br(g(cmf, "logq", "med_abs"))); M("cAlTrintaForte", pc(g(cmf, "alpha_deg", "frac_lt30")))
M("cSAuc", br(g(cm, "sign", "auc"))); M("cAls", br(g(cm, "abs_logs", "med_abs"), 3))
for nome, par in (("Tz", "t0"), ("Uz", "u0"), ("Te", "tE")):
    M("cBeta" + nome, br(g(CRV, "grupos", "todos", par, "beta"))); M("cMelhor" + nome, pc(g(CRV, "grupos", "todos", par, "frac_rede_melhor")))
# proveniência
M("gitCommit", (g(MAN, "git_commit") or "")[:7] or None); M("melhorEpoca", str(g(META, "best_epoch") + 1) if ok(g(META, "best_epoch")) else None)

with open(OUT / "resultados/numeros.tex", "w", encoding="utf-8") as f:
    f.write("% GERADO por scripts/16_latex.py a partir dos JSON em models/. NÃO EDITE À MÃO: rode o script de novo.\n")
    for k, v in MAC.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v if v is not None else chr(92) + 'pendente{' + k + '}'}}}\n")

# ------------------------------------------------------------------------------------------------------------------
# Tabelas
# ------------------------------------------------------------------------------------------------------------------
PEND = r"\pendente{--}"
c = lambda v: v if v is not None else PEND


def tab(nome, colunas, cab, linhas):
    with open(OUT / f"resultados/{nome}.tex", "w", encoding="utf-8") as f:
        f.write("% GERADO por scripts/16_latex.py. NÃO EDITE À MÃO.\n\\begin{tabular}{" + colunas + "}\n\t\\toprule\n")
        for l in cab: f.write("\t" + l + "\n")
        f.write("\t\\midrule\n")
        for l in linhas: f.write("\t" + (l if isinstance(l, str) else " & ".join(c(x) for x in l) + r" \\") + "\n")
        f.write("\t\\bottomrule\n\\end{tabular}\n")


par2 = lambda s, k, n: None if g(s, k, "med_abs") is None else f"{br(g(s, k, 'med_abs'), n)} ({br(g(s, k, 'sigma68'), n)})"
tab("tab_erros", "lccc", [rf"Parâmetro & Todos ($n = {c(mil(g(m, 'n')))}$) & Excesso de $\chi^2 > 100$ ($n = {c(mil(g(mf, 'n')))}$) & Referência \\"], [
    [r"$\tz$ (dias)", par2(m, "t0", 2), par2(mf, "t0", 2), br(REF["t0"], 2)],
    [r"$\uz$", par2(m, "u0", 3), par2(mf, "u0", 3), br(REF["u0"], 3)],
    [r"$\tE$ (\%)", par2(m, "tE_pct", 1), par2(mf, "tE_pct", 1), br(REF["tE"], 1)],
    [r"$\log q$ (dex)", par2(m, "logq", 2), par2(mf, "logq", 2), br(REF["logq"], 2)],
    [r"$\log s$ (dex)", par2(m, "logs", 2), par2(mf, "logs", 2), br(REF["logs"], 2)],
    [r"$\alpha$ (graus)", None if g(m, "alpha_deg") is None else f"{br(m['alpha_deg']['med_abs'], 0)} ({br(m['alpha_deg']['sigma68'], 0)})",
     None if g(mf, "alpha_deg") is None else f"{br(mf['alpha_deg']['med_abs'], 0)} ({br(mf['alpha_deg']['sigma68'], 0)})", "90"],
    [r"Sinal de $\log s$: acerto", (pc(g(m, "sign", "acc")) or PEND) + r"\%", (pc(g(mf, "sign", "acc")) or PEND) + r"\%", r"50\%"],
    [r"Sinal de $\log s$: AUC", br(g(m, "sign", "auc")), br(g(mf, "sign", "auc")), "0,50"]])

rr = lambda grp, k: None if g(r2k("final", grp), k) is None else f"{br(r2k('final', grp)[k]['r2'])} {ic(r2k('final', grp)[k].get('ic95'))}"
tab("tab_r2", "lccc", [r"Parâmetro & Todos & Excesso de $\chi^2 > 100$ & Acaso \\"], [
    [r"$\tz$", rr("todos", "t0"), rr("forte", "t0"), "0"], [r"$\uz$", rr("todos", "u0"), rr("forte", "u0"), "0"],
    [r"$\log \tE$", rr("todos", "log tE"), rr("forte", "log tE"), "0"], [r"$\tE$ (linear)", rr("todos", "tE (linear)"), rr("forte", "tE (linear)"), "0"],
    [r"$\log q$", rr("todos", "log q"), rr("forte", "log q"), "0"], [r"$|\log s|$", rr("todos", "|log s|"), rr("forte", "|log s|"), "0"],
    [r"$\alpha$: erro $< 30^\circ$", (pc(g(m, "alpha_deg", "frac_lt30")) or PEND) + r"\%", (pc(g(mf, "alpha_deg", "frac_lt30")) or PEND) + r"\%", r"17\%"]])


def lp(nome, par, n):
    l = [nome]
    for grp in ("todos", "excesso>100"):
        l += [br(rv(grp, par, "erro_mediano_pspl"), n), br(rv(grp, par, "erro_mediano_rede"), n), pc(rv(grp, par, "frac_rede_melhor")),
              None if rv(grp, par, "beta") is None else f"{br(rv(grp, par, 'beta'))} {ic(rv(grp, par, 'ic95_beta'))}"]
    return l


tab("tab_pspl", "lcccccccc", [r" & \multicolumn{4}{c}{Todos} & \multicolumn{4}{c}{Excesso de $\chi^2 > 100$} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
    r"Parâmetro & PSPL & Rede & \% & $\beta$ & PSPL & Rede & \% & $\beta$ \\"],
    [lp(r"$\tz$ (dias)", "t0", 3), lp(r"$\uz$", "u0", 3), lp(r"$\log \tE$ (dex)", "tE", 3)])


def faixa(runs, grp, f, n, pct=False):
    """média (mín–máx) entre sementes de f(métricas)."""
    v = [f(MET[r][grp]) for r in runs if MET.get(r) is not None]
    v = [100 * x if pct else x for x in v if ok(x)]
    if not v: return None
    return br(v[0], n) if len(v) == 1 else f"{br(sum(v) / len(v), n)} ({br(min(v), n)}--{br(max(v), n)})"


LINV = [(r"$\tz$ (dias)", lambda s: s["t0"]["med_abs"], 3, False), (r"$\uz$", lambda s: s["u0"]["med_abs"], 3, False),
        (r"$\tE$ (\%)", lambda s: s["tE_pct"]["med_abs"], 1, False), (r"$\log q$ (dex)", lambda s: s["logq"]["med_abs"], 2, False),
        (r"$|\log s|$ (dex)", lambda s: s["abs_logs"]["med_abs"], 3, False), (r"$\alpha$ (graus)", lambda s: s["alpha_deg"]["med_abs"], 0, False),
        (r"$\alpha$: erro $< 30^\circ$ (\%)", lambda s: s["alpha_deg"]["frac_lt30"], 0, True), (r"Sinal de $\log s$: AUC", lambda s: s["sign"]["auc"], 2, False)]
nA, nF = sum(MET[r] is not None for r in ABL), sum(MET[r] is not None for r in SEM)
for nome, grp in (("tab_versoes", "all"), ("tab_versoes_forte", "exc>100")):
    tab(nome, "lccc", [rf"Parâmetro & Sem $u(\tau)$ ({nA} sem.) & Com $u(\tau)$ (1 sem.) & Com $u(\tau)$ e $|s - 1/s|$ ({nF} sem.) \\"],
        [[t, faixa(ABL, grp, f, n, p), faixa(["abl_u"], grp, f, n, p), faixa(SEM, grp, f, n, p)] for t, f, n, p in LINV])
tab("tab_sementes", "l" + "c" * max(nF, 1), ["Parâmetro & " + " & ".join(f"Semente {i}" for i in range(max(nF, 1))) + r" \\"],
    [[t] + [None if MET[r] is None else f"{br((100 if p else 1) * f(MET[r]['all']), n)} / {br((100 if p else 1) * f(MET[r]['exc>100']), n)}" for r in SEM if MET[r] is not None or nF == 0]
     for t, f, n, p in LINV] +
    [[r"$\beta$ de $\uz$"] + [br(rv("todos", "u0", "beta", r)) + " / " + br(rv("excesso>100", "u0", "beta", r)) if ok(rv("todos", "u0", "beta", r)) else None for r in SEM if MET[r] is not None],
     [r"$\beta$ de $\tE$"] + [br(rv("todos", "tE", "beta", r)) + " / " + br(rv("excesso>100", "tE", "beta", r)) if ok(rv("todos", "tE", "beta", r)) else None for r in SEM if MET[r] is not None]])


def ld(nome, D_, det):
    return [nome, br(g(D_, "AUC_rede" if det == "rede" else "AUC_excesso_chi2")), (pc(d5(D_, det, "completeza_total")) or PEND) + r"\%"] + \
           [(pc(d5(D_, det, "completeza_por_excesso", b)) or PEND) + r"\%" for b in ("<10", "10-100", "100-1000", ">1000")]


tab("tab_deteccao", "lcccccc", [r" & & & \multicolumn{4}{c}{Completeza por faixa de excesso de $\chi^2$} \\", r"\cmidrule(lr){4-7}",
    r"Detector & AUC & Completeza & $<10$ & $10$--$100$ & $100$--$1000$ & $>1000$ \\"],
    [r"\multicolumn{7}{l}{\textit{Todos os eventos com companheira ($10^{-4} \leq q \leq 10^{-1}$; $n = " + c(mil(g(DET, "n_com_planeta"))) + r"$)}} \\",
     ld("Rede", DET, "rede"), ld(r"Excesso de $\chi^2$", DET, "excesso_chi2"), r"\midrule",
     r"\multicolumn{7}{l}{\textit{Regime planetário ($q < 0{,}03$; $n = " + c(mil(g(DETP, "n_com_planeta"))) + r"$)}} \\",
     ld("Rede", DETP, "rede"), ld(r"Excesso de $\chi^2$", DETP, "excesso_chi2")])

l1 = lambda k: [br(g(L1, k, "R2", "t0"), 3), br(g(L1, k, "R2", "u0"), 3), br(g(L1, k, "R2", "tE"), 3), br(g(L1, k, "t0_med_dias"), 3), br(g(L1, k, "u0_med"), 3), br(g(L1, k, "tE_med_pct"), 1)]
tab("tab_1l1s", "lcccccc", [r" & \multicolumn{3}{c}{$R^2$} & \multicolumn{3}{c}{Erro mediano} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
    r"Estimador & $\tz$ & $\uz$ & $\tE$ & $\tz$ (dias) & $\uz$ & $\tE$ (\%) \\"],
    [["Rede (todos)"] + l1("rede_todos"), ["Rede (classificados 1L1S)"] + l1("rede_classificados_simples"), ["Ajuste 1L1S"] + l1("ajuste_1L1S_todos")])

NOME = {"oraculo": "Oráculo (partida verdadeira)", "rede_1ramo": "Rede, um ramo", "rede_2ramos": "Rede, dois ramos",
        "cega_mesmo_orcamento": "Cega, mesmo orçamento", "cega_8": "Cega, oito partidas"}
kn_ = lambda s, crit: None if be(s, crit, "k") is None else f"{be(s, crit, 'k')} {ic(be(s, crit, 'ic95'), pc, 0, '--')}"
tab("tab_benchmark", "lcccccc", [r"Estratégia & Ajustes & Convergência & Solução física & $\chi^2/N < 1{,}5$ & Tempo (s) & Custo (s) \\"],
    [[NOME[s], br(be(s, "n_ajustes_medio"), 1), kn_(s, "convergencia"), kn_(s, "fisica"), str(be(s, "chi2N_menor_1_5", "k")) if ok(be(s, "chi2N_menor_1_5", "k")) else None,
      sig(be(s, "tempo_mediano_s"), 2), br(be(s, "custo_por_sucesso_conv_s"), 0)] for _, s in EST])
tab("tab_benchmark_excesso", "lccccc", [r"Excesso de $\chi^2$ & $n$ & Oráculo & Rede, dois ramos & Cega, mesmo orç. & Cega, oito \\"],
    [[r"$< 10^2$" if b == "<100" else r"$10^2$--$10^3$" if b == "100-1000" else r"$10^3$--$10^5$" if b == "1000-1e5" else r"$> 10^5$", str(cx(b, "n")) if ok(cx(b, "n")) else None] +
     [(pc(cx(b, s)) + r"\%") if ok(cx(b, s)) else "--" for s in ("oraculo", "rede_2ramos", "cega_mesmo_orcamento", "cega_8")] for b in ("<100", "100-1000", "1000-1e5", ">1e5")])


def lo(e):
    o = OG[e]; mq = lambda v: None if sci(v) is None else f"${sci(v)}$"
    red = [" & Rede", mq(g(o, "rede", "q")), None if g(o, "rede") is None else f"{br(o['rede']['s_wide'])} ou {br(o['rede']['s_close'])}", "--", "--", "--"]
    det = g(o, "ajuste", "detectado"); dc = g(o, "ajuste", "dchi2_pspl"); dcs = None if not ok(dc) else (mil(dc) if dc >= 100 else br(dc, 0))
    aj = ([" & Ajuste", mq(g(o, "ajuste", "q")), br(g(o, "ajuste", "s")), sig(g(o, "ajuste", "u0"), 2), sig(g(o, "ajuste", "tE"), 2), dcs] if det or det is None
          else r" & Ajuste & \multicolumn{4}{c}{não detectado} & " + c(dcs) + r" \\")
    return [" & Lente simples", "--", "--", sig(g(o, "pspl", "u0"), 2), sig(g(o, "pspl", "tE"), 2), "--"], red, aj


a1, a2, a3 = lo("2012-BLG-0406"); b1, b2, b3 = lo("2016-BLG-1195")
fix = lambda l: l if isinstance(l, str) else l
tab("tab_ogle", "llccccc", [r"Evento & & $q$ & $s$ & $\uz$ & $\tE$ (d) & $\Delta\chi^2$ \\"],
    [r"OGLE-2012-BLG-0406 & Literatura & $6{,}3 \times 10^{-3}$ & 1,31 & 0,51 & 64 & -- \\", a1, a2, a3, r"\midrule",
     r"OGLE-2016-BLG-1195 & Literatura & $4{,}2 \times 10^{-5}$ & 1,07 ou 1,00 & 0,051 & 10,2 & -- \\", b1, b2, b3])

# ------------------------------------------------------------------------------------------------------------------
# Figuras (PNG -> JPG com os nomes usados no texto)
# ------------------------------------------------------------------------------------------------------------------
FIGS = {"fig_r2_final.png": "fig_previsto_verdadeiro.jpg", "fig_rede_vs_pspl_final.png": "fig_rede_vs_pspl.jpg",
        "fig_erro_vs_excesso_final.png": "fig_erro_vs_excesso.jpg", "fig_s_analise_final.png": "fig_s_analise.jpg",
        "fig_deteccao_final_det.png": "fig_deteccao.jpg", "fig_1l1s_final_det.png": "fig_1l1s.jpg", "fig_benchmark_v2.png": "fig_benchmark.jpg",
        "ogle_2012-BLG-0406.png": "ogle_2012-BLG-0406.jpg", "ogle_2016-BLG-1195.png": "ogle_2016-BLG-1195.jpg"}
try:
    from PIL import Image
    for src, dst in FIGS.items():
        if (FIG / src).exists():
            im = Image.open(FIG / src).convert("RGB"); im.thumbnail((2600, 2600)); im.save(OUT / "figuras" / dst, quality=88)
        else:
            FALTAM.append("figures/" + src)
except ImportError:
    print("AVISO: instale o Pillow (pip install pillow) para converter as figuras; copie os PNG à mão por enquanto.")

pend = [k for k, v in MAC.items() if v is None]
print(f"{len(MAC)} números em {OUT / 'resultados/numeros.tex'}; tabelas em {OUT / 'resultados'}; figuras em {OUT / 'figuras'}")
if FALTAM: print("Arquivos não encontrados (os números correspondentes ficam pendentes):", ", ".join(sorted(set(FALTAM))))
if pend: print(f"{len(pend)} números pendentes:", " ".join(pend))
