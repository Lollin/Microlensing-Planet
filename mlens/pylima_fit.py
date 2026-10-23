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
