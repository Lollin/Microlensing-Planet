"""Ajuste USBL (2L1S) com pyLIMA a partir de arrays (t em JD, fluxo normalizado, erro). Testado com pyLIMA 1.9.8."""
import time, warnings, numpy as np
warnings.filterwarnings("ignore")
from pyLIMA import event, telescopes
from pyLIMA.models import USBL_model
from pyLIMA.fits import TRF_fit

MAG0 = 18.0   # ponto-zero arbitrário (só reescala fluxo <-> mag)


def make_event(data_jd, name="evt", ra=270.0, dec=-30.0):
    ev = event.Event(ra=ra, dec=dec); ev.name = name
    for site, (t, f, s) in data_jd.items():
        ok = (f > 0)
        mag = MAG0 - 2.5 * np.log10(f[ok]); emag = 1.0857 * s[ok] / f[ok]
        ev.telescopes.append(telescopes.Telescope(name=site, camera_filter="I", lightcurve=np.c_[t[ok], mag, emag],
                             lightcurve_names=["time", "mag", "err_mag"], lightcurve_units=["JD", "mag", "mag"]))
    ev.find_survey(next(iter(data_jd))); ev.check_event()
    return ev


def fit_usbl(data_jd, guess, rho=1e-3, loss="chi2"):
    """guess = [t0_JD, u0, tE, s, q, alpha]; retorna dict(params, chi2, tempo)."""
    ev = make_event(data_jd); model = USBL_model.USBLmodel(ev)
    f = TRF_fit.TRFfit(model, loss_function=loss)
    f.model_parameters_guess = [guess[0], guess[1], guess[2], rho, guess[3], guess[4], guess[5]]
    t0 = time.time(); f.fit()
    if "best_model" not in f.fit_results:
        raise RuntimeError("pyLIMA recusou o ponto de partida (parâmetro fora dos limites); ajuste não realizado")
    b = f.fit_results["best_model"]
    return dict(t0=b[0], u0=b[1], tE=b[2], rho=b[3], s=b[4], q=b[5], alpha=b[6], chi2=float(f.fit_results["chi2"]), seconds=time.time() - t0)


def blind_multistart(data_jd, base, s_grid=(0.8, 1.3), alpha_grid=(0.0, 1.57, 3.14, 4.71), q0=1e-3):
    """Busca 'tradicional' simplificada: várias partidas em (s, alpha), melhor chi2. base = [t0_JD, u0, tE]."""
    best, tot = None, 0.0
    for s in s_grid:
        for a in alpha_grid:
            try: r = fit_usbl(data_jd, [*base, s, q0, a])
            except Exception as e: continue
            tot += r["seconds"]
            if best is None or r["chi2"] < best["chi2"]: best = r
    if best: best["seconds"] = tot
    return best


# ----------------------------------------------------------------------------------------------------------------
# Ajuste em dois ramos (close / wide), mais o ressonante quando couber.
# A rede estima a magnitude x = |s − 1/s|, que é a mesma para s e 1/s. Os dois candidatos são
#   s_w = (x + sqrt(x² + 4)) / 2   (raiz positiva de s² − x s − 1 = 0)   e   s_c = 1 / s_w.
# A topologia de cada candidato é obtida numericamente contando as cáusticas (VBMicrolensing):
#   3 = próxima (close), 1 = ressonante, 2 = larga (wide).
# Ajusta-se a partir de cada candidato e o Δχ² decide. Se |Δχ²| for pequeno, as duas soluções são degeneradas.
# ----------------------------------------------------------------------------------------------------------------
def topologia(s, q):
    import VBMicrolensing
    if not hasattr(topologia, "_v"):
        topologia._v = VBMicrolensing.VBMicrolensing()
    return {3: "close", 1: "ressonante", 2: "wide"}.get(len(topologia._v.Caustics(float(s), float(q))), "?")


def candidatos_s(abs_logs, q):
    """|log s| previsto pela rede -> lista de (rótulo, s) com close, wide e, se algum cair na zona ressonante, s = 1."""
    s_w = 10 ** abs(abs_logs); s_c = 1 / s_w
    cands = [("wide", s_w), ("close", s_c)]
    if "ressonante" in (topologia(s_w, q), topologia(s_c, q)):
        cands.append(("ressonante", 1.0))
    return cands


def fit_branches(data_jd, base, abs_logs, q, alpha, rho=1e-3, limiar=10.0):
    """Ajusta a partir de cada candidato de s. base = [t0_JD, u0, tE].
    Retorna dict com o melhor ajuste, todos os ajustes, a topologia do melhor e Δχ² entre o melhor e o melhor do outro ramo."""
    fits, tot = [], 0.0
    for lab, s0 in candidatos_s(abs_logs, q):
        try:
            r = fit_usbl(data_jd, [*base, s0, q, alpha], rho=rho)
        except Exception:
            continue
        r["ramo_partida"] = lab; r["topologia"] = topologia(r["s"], r["q"]); fits.append(r); tot += r["seconds"]
    if not fits:
        return None
    fits.sort(key=lambda r: r["chi2"]); best = dict(fits[0]); best["seconds"] = tot; best["n_ajustes"] = len(fits)
    outros = [r for r in fits[1:] if r["topologia"] != best["topologia"]]
    best["dchi2_outro_ramo"] = float(outros[0]["chi2"] - best["chi2"]) if outros else float("nan")
    best["ramo_definido"] = bool(outros and best["dchi2_outro_ramo"] > limiar)
    best["todos"] = [{k: r[k] for k in ("ramo_partida", "topologia", "s", "q", "chi2")} for r in fits]
    return best


def fit_pspl(data_jd, guess):
    """Ajuste de lente simples (PSPL) com o pyLIMA, para comparar χ² na mesma escala do ajuste binário. guess = [t0_JD, u0, tE]."""
    from pyLIMA.models import PSPL_model
    ev = make_event(data_jd); model = PSPL_model.PSPLmodel(ev)
    f = TRF_fit.TRFfit(model); f.model_parameters_guess = list(guess); f.fit()
    if "best_model" not in f.fit_results:
        raise RuntimeError("pyLIMA recusou o ponto de partida do ajuste PSPL")
    b = f.fit_results["best_model"]
    return dict(t0=b[0], u0=b[1], tE=b[2], chi2=float(f.fit_results["chi2"]))
