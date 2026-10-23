"""Geração paralela de dataset. Saída: data/processed/*.npz (X: N x 2 x N_BINS, y: N x 7)."""
import numpy as np
from multiprocessing import Pool
from . import config as C
from .simulate import simulate_event, to_grid, params_to_vec


def _one(seed):
    rng = np.random.default_rng(seed)
    p, data = simulate_event(rng)
    return to_grid(data), params_to_vec(p)


def build(n, seed0=C.SEED, workers=4):
    with Pool(workers) as pool:
        out = pool.map(_one, range(seed0, seed0 + n), chunksize=20)
    X = np.stack([o[0] for o in out]); y = np.stack([o[1] for o in out])
    return X, y
