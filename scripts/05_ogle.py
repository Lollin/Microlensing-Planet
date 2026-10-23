"""Eventos reais do OGLE: baixa a fotometria (EWS), calcula features, roda a CNN e o ajuste pyLIMA, compara.
Ex.:   python scripts/05_ogle.py --event 2019-BLG-0123 --model models/cnn_2l1s.pt
       python scripts/05_ogle.py --file meu_evento/phot.dat --model models/cnn_2l1s.pt          (arquivo baixado à mão)
ATENÇÃO: a política do OGLE pede contato com a equipe para uso em PUBLICAÇÃO dos dados EWS. Confirme as URLs no site do OGLE."""
import sys, pathlib, argparse, io, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mlens.features import featurize, pspl_fit
from mlens.pylima_fit import fit_usbl

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
    ap = argparse.ArgumentParser(); ap.add_argument("--event"); ap.add_argument("--file"); ap.add_argument("--model", default="models/cnn_2l1s.pt")
    ap.add_argument("--no-nn", action="store_true"); a = ap.parse_args()
    text = open(a.file).read() if a.file else download(a.event); t, m, e = load_phot(text); tref, data = to_model_frame(t, m, e)
    X, sc, d2, fit = featurize(data); print(f"N={len(t)}  PSPL: u0={fit['u0']:.3f} tE={fit['tE']:.1f} d  Δχ²≈{d2:.0f}  (treino: tE 10–50 d)")
    guess = [fit["t0"] + tref, fit["u0"], fit["tE"], 1.0, 1e-3, 0.0]
    if not a.no_nn:
        from mlens.torch_cnn import predict
        P = predict(a.model, X[None], sc[None]); guess[3:] = [10 ** float(P["logs"][0]), 10 ** float(P["logq"][0]), float(P["alpha"][0])]
        print(f"CNN: s={guess[3]:.2f}  q={guess[4]:.2e}  α={guess[5]:.2f} rad")
    jd = {"OGLE": (t, 10 ** (-0.4 * (m - 18.0)) / np.percentile(10 ** (-0.4 * (m - 18.0)), 20), (10 ** (-0.4 * (m - 18.0)) / np.percentile(10 ** (-0.4 * (m - 18.0)), 20)) * e / 1.0857)}
    r = fit_usbl(jd, guess); print("pyLIMA (init. da CNN):", {k: round(float(v), 4) for k, v in r.items()})
    plt.figure(figsize=(9, 4)); plt.errorbar(t - 2450000, m, e, fmt=".", ms=3, alpha=.5); plt.gca().invert_yaxis()
    plt.xlabel("HJD − 2450000"); plt.ylabel("I (mag)"); plt.savefig("figures/ogle_evento.png", dpi=150, bbox_inches="tight"); print("figura: figures/ogle_evento.png")
