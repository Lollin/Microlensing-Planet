"""Dataset com features de resíduo. Saída: X (N,3,256), sc (N,5), y (N,7), dchi2 (N,)"""
import numpy as np
from multiprocessing import Pool
from . import config as C
from .features import featurize
from .simulate import simulate_event, params_to_vec, sample_params


def _one(args):
    seed, engine, frac = args
    rng = np.random.default_rng(seed)
    if engine == "pylima":
        from .simulate_pylima import simulate_event_pylima
        p, data, _ = simulate_event_pylima(rng)
    else:
        p, data = simulate_event(rng, sample_params(rng, frac))
    X, sc, d2, _ = featurize(data)
    return X, sc, params_to_vec(p), d2


def build(n, seed0=C.SEED, workers=4, engine="vbm", frac=None):
    with Pool(workers) as pool:
        out = pool.map(_one, [(s, engine, frac) for s in range(seed0, seed0 + n)], chunksize=10)
    return [np.stack([o[i] for o in out]) for i in range(4)]
