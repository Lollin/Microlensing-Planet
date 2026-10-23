"""Simulação 2L1S com pyLIMA (modelo USBL). Mesmo formato de saída de simulate.py:
   (p, data) com data = {sítio: (t_rel_dias, fluxo_normalizado, erro)}  -> compatível com features.py / to_grid.
   Testado com pyLIMA 1.9.8 (a API muda entre versões!)."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from pyLIMA.simulations import simulator
from pyLIMA.models import USBL_model
from . import config as C

JD0 = 2460000.0
# (lat, lon, altitude m) — Las Campanas, Sutherland, Siding Spring
SITES_GEO = {"Chile": (-29.0, -70.7, 2400), "Africa do Sul": (-32.4, 20.8, 1760), "Australia": (-31.3, 149.1, 1165)}


def build_event(rng, p, mag_s=(18.0, 20.5), mag_b=(19.0, 22.0)):
    """Os instantes vêm de simulate.observation_times (rápido, mesmo cenário do VBM); o pyLIMA calcula só magnificação + ruído.
    (O agendador astronômico do próprio pyLIMA custa ~13 s/evento; use-o só para conjuntos pequenos.)"""
    from .simulate import observation_times
    ev = simulator.simulate_a_microlensing_event(name="sim", ra=270, dec=-30)
    for name, t in observation_times(rng).items():
        lat, lon, alt = SITES_GEO[name]
        tel = simulator.simulate_a_telescope(name, time_start=JD0 + C.T_MIN, time_end=JD0 + C.T_MAX, sampling=1.0,
                location="Earth", camera_filter="I", uniform_sampling=True, timestamps=list(JD0 + t),
                altitude=alt, longitude=lon, latitude=lat, astrometry=False)
        ev.telescopes.append(tel)
    model = USBL_model.USBLmodel(ev)                       # blend_flux_parameter='fblend' (padrão): [fs, fb] por telescópio
    fluxes = []
    for tel in ev.telescopes:                              # fluxos de fonte/blend por telescópio (mesma estrela => mesma mag)
        fluxes += simulator.simulate_fluxes_parameters([tel], source_magnitude=list(mag_s), blend_magnitude=list(mag_b))
    return ev, model, fluxes


def p_to_list(p):
    return [JD0 + p["t0"], p["u0"], p["tE"], C.RHO, 10 ** p["logs"], 10 ** p["logq"], p["alpha"]]


def simulate_event_pylima(rng, p=None):
    from .simulate import sample_params
    p = p or sample_params(rng)
    np.random.seed(int(rng.integers(0, 2**31 - 1)))        # pyLIMA usa np.random global
    ev, model, fl = build_event(rng, p)
    # mesmo fonte/blend em todos os telescópios -> repete o 1º par
    fs, fb = fl[0], fl[1]
    pars = p_to_list(p) + [x for _ in ev.telescopes for x in (fs, fb)]
    simulator.simulate_lightcurve(model, model.compute_pyLIMA_parameters(pars), add_noise=True)
    data = {}
    for tel in ev.telescopes:
        lc = tel.lightcurve
        base = fs + fb
        data[tel.name] = (np.asarray(lc["time"].value if hasattr(lc["time"], "value") else lc["time"], float) - JD0,
                          np.asarray(lc["flux"], float) / base, np.asarray(lc["err_flux"], float) / base)
    return p, data, ev
