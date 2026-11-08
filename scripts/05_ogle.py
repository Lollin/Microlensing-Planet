"""Eventos reais do OGLE: baixa a fotometria (EWS), calcula features, roda a CNN e o ajuste pyLIMA, compara.
Versão 3: (1) reescala os erros pela dispersão entre pontos consecutivos (não depende de modelo); (2) usa só a janela |τ| < 4, a mesma vista
pela rede; (3) ajusta em dois ramos (close/wide, + ressonante); (4) critério de detecção: Δχ² = χ²(lente simples) − χ²(binária).
Ex.:   python scripts/05_ogle.py --event 2019-BLG-0123 --model models/cnn_2l1s.pt
       python scripts/05_ogle.py --file meu_evento/phot.dat --model models/cnn_2l1s.pt          (arquivo baixado à mão)
ATENÇÃO: a política do OGLE pede contato com a equipe para uso em PUBLICAÇÃO dos dados EWS. Confirme as URLs no site do OGLE."""
import sys, pathlib, argparse, io, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mlens.features import featurize, pspl_fit
from mlens.pylima_fit import fit_usbl, fit_pspl, fit_branches

URLS = ["https://www.astrouw.edu.pl/ogle/ogle4/ews/{y}/blg-{n:04d}/phot.dat",
        "http://www.astrouw.edu.pl/ogle/ogle4/ews/{y}/blg-{n:04d}/phot.dat"]


def download(event):
    import requests
    y, _, n = event.split("-")                       # '2019-BLG-0123'
    for u in URLS:
        r = requests.get(u.format(y=y, n=int(n)), timeout=60)
        if r.ok and len(r.text) > 100: return r.text
    raise RuntimeError("Não consegui baixar; abra a página do evento no site do OGLE e passe o arquivo com --file.")


def load_phot(text):
    """phot.dat do EWS: colunas  HJD  I_mag  I_err  seeing  sky  (HJD completo ou HJD-2450000)."""
    A = np.loadtxt(io.StringIO(text), usecols=(0, 1, 2)); t, m, e = A.T
    t = np.where(t < 2.4e6, t + 2450000.0, t); ok = (e > 0) & (e < 0.5) & (m > 10) & (m < 25)
    return t[ok], m[ok], e[ok]


def to_model_frame(t, m, e):
    F = 10 ** (-0.4 * (m - 18.0)); base = np.percentile(F, 20); f = F / base; s = f * e / 1.0857     # fluxo ~1 fora do evento
    pk = np.argmax(np.convolve(f, np.ones(5) / 5, "same")); tref = t[pk] - 100.0                     # centra o pico na janela de treino
    return tref, {"OGLE": (t - tref, f, s)}


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument("--event"); ap.add_argument("--file"); ap.add_argument("--model", default="models/cnn_2l1s.pt")
    ap.add_argument("--no-nn", action="store_true"); ap.add_argument("--janela", type=float, default=4.0, help="meia-largura da janela em unidades de tE")
    ap.add_argument("--limiar", type=float, default=100.0, help="Δχ² mínimo para considerar a anomalia binária detectada")
    a = ap.parse_args()
    text = open(a.file).read() if a.file else download(a.event); t, m, e = load_phot(text); tref, data = to_model_frame(t, m, e)
    tt, f, s = data["OGLE"]
    # fator de reescala dos erros sem depender de modelo: diferenças entre pontos consecutivos.
    # Se o ruído verdadeiro é k·σ, então (f_i+1 − f_i)² / (σ_i² + σ_i+1²) segue k²·χ²(1), cuja mediana é 0,455·k².
    d = np.diff(f); v = s[1:] ** 2 + s[:-1] ** 2
    k = max(float(np.sqrt(np.median(d ** 2 / v) / 0.4549)), 1.0); s = s * k
    fit0 = pspl_fit(tt, f, s, robust=True); tau = (tt - fit0["t0"]) / fit0["tE"]     # pré-ajuste robusto: só para a janela
    w = np.abs(tau) < a.janela
    X, sc, d2, fit = featurize({"OGLE": (tt[w], f[w], s[w])})
    print(f"N total={len(t)}  na janela={int(w.sum())}  fator de reescala dos erros k={k:.2f}")
    print(f"Lente simples: u0={fit['u0']:.3f}  tE={fit['tE']:.1f} d  excesso de χ²={d2:.0f}  (treino: tE 10–50 d)")
    jd = {"OGLE": (tt[w] + tref, f[w], s[w])}; base = [fit["t0"] + tref, fit["u0"], fit["tE"]]
    try:
        c_pspl = fit_pspl(jd, base)["chi2"]
    except Exception as ex:
        c_pspl = float(fit["chi2"]); print("PSPL do pyLIMA falhou, usando o ajuste interno:", ex)
    if a.no_nn:
        r = fit_branches(jd, base, 0.1, 1e-3, 0.0)
    else:
        from mlens.torch_cnn import predict
        P = predict(a.model, X[None], sc[None]); q0, al0, al_s = 10 ** float(P["logq"][0]), float(P["alpha"][0]), float(P["abs_logs"][0])
        pp = f"  p(planeta)={float(P['p_planet'][0]):.2f}" if "p_planet" in P else ""
        print(f"Rede: |log s|={al_s:.3f} (s={10 ** al_s:.2f} ou {10 ** -al_s:.2f})  q={q0:.2e}  α={al0:.2f} rad  p(wide)={float(P['p_wide'][0]):.2f}{pp}")
        r = fit_branches(jd, base, al_s, q0, al0)
    if r is None:
        print("Nenhum ajuste binário convergiu.")
    else:
        for x in r["todos"]:
            print(f"  ajuste partindo de {x['ramo_partida']:<10}: s={x['s']:.3f}  q={x['q']:.2e}  topologia={x['topologia']:<10}  χ²={x['chi2']:.1f}")
        dc = c_pspl - r["chi2"]; N = int(w.sum())
        print(f"Melhor: s={r['s']:.3f}  q={r['q']:.2e}  u0={r['u0']:.3f}  tE={r['tE']:.1f}  rho={r['rho']:.4f}  topologia={r['topologia']}  χ²/N={r['chi2'] / N:.2f}")
        print(f"Δχ²(lente simples − binária) = {dc:.1f}  ->  {'ANOMALIA BINÁRIA DETECTADA' if dc > a.limiar else 'sem detecção (abaixo do limiar ' + str(a.limiar) + ')'}")
        if r["chi2"] / N > 2:
            print("O melhor ajuste binário ainda é ruim (χ²/N > 2): não convergiu; o ramo (close/wide) não pode ser decidido.")
        elif r["ramo_definido"]:
            print(f"Ramo definido: {r['topologia']}; o outro ramo é pior por Δχ² = {r['dchi2_outro_ramo']:.1f}")
        elif np.isnan(r["dchi2_outro_ramo"]):
            print(f"Todos os ajustes terminaram na mesma topologia ({r['topologia']}).")
        else:
            print(f"Ramo NÃO definido (degenerescência close/wide): Δχ² entre ramos = {r['dchi2_outro_ramo']:.1f}")
    nome = (a.event or pathlib.Path(a.file).stem).replace("/", "_")
    pathlib.Path("figures").mkdir(exist_ok=True)
    plt.figure(figsize=(9, 4)); plt.errorbar(t - 2450000, m, e * k, fmt=".", ms=3, alpha=.5); plt.gca().invert_yaxis()
    plt.xlim((base[0] - 2450000) - a.janela * fit["tE"], (base[0] - 2450000) + a.janela * fit["tE"])
    plt.xlabel("HJD − 2450000"); plt.ylabel("I (mag)"); plt.title(f"OGLE-{nome}")
    out = f"figures/ogle_{nome}.png"; plt.savefig(out, dpi=150, bbox_inches="tight"); print("figura:", out)
