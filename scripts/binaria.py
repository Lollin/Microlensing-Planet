"""Lente binária (2L1S) numérica: cáusticas (quártica por φ), imagens (quíntica de Witt), magnificação e topologia.
Módulo usado por 08_degenerescencia_s.py. Teste rápido: python scripts/binaria.py"""
import numpy as np, sympy as sp

def posicoes(s, q):
    m1, m2 = 1/(1+q), q/(1+q)
    z1, z2 = -m2*s, m1*s            # centro de massa na origem, eixo real
    return m1, m2, z1, z2

def caustica(s, q, n=2000):
    m1, m2, a, b = posicoes(s, q)
    zc, zeta = [], []
    for phi in np.linspace(0, 2*np.pi, n, endpoint=False):
        e = np.exp(1j*phi)
        c = [1, -2*(a+b), (a+b)**2 + 2*a*b - e,
             -2*a*b*(a+b) + 2*e*(m1*b + m2*a), a*a*b*b - e*(m1*b*b + m2*a*a)]
        w = np.roots(c)                       # w = z barra
        z = np.conj(w)
        zc += list(z); zeta += list(z - m1/(w - a) - m2/(w - b))
    return np.array(zc), np.array(zeta)

# coeficientes do polinomio de grau 5 gerados simbolicamente (uma vez)
_z, _zt, _m1, _m2, _a, _b = sp.symbols('z zeta_c m1 m2 a b')   # zeta_c = conj(zeta)
_zeta = sp.Symbol('zeta')
_zbar = _zt + _m1/(_z - _a) + _m2/(_z - _b)                     # equacao conjugada
_expr = sp.together(_z - _m1/(_zbar - _a) - _m2/(_zbar - _b) - _zeta)
_num = sp.Poly(sp.numer(_expr), _z)
_coef = sp.lambdify((_m1, _m2, _a, _b, _zeta, _zt), _num.all_coeffs(), "numpy")

def imagens(zeta, s, q, tol=1e-6):
    m1, m2, a, b = posicoes(s, q)
    c = np.array(_coef(m1, m2, a, b, zeta, np.conj(zeta)), dtype=complex)
    z = np.roots(c)
    res = np.abs(z - m1/(np.conj(z) - a) - m2/(np.conj(z) - b) - zeta)
    z = z[res < tol]                                   # descarta raizes espurias
    dJ = 1 - np.abs(m1/(np.conj(z) - a)**2 + m2/(np.conj(z) - b)**2)**2
    return z, 1/dJ                                     # posicoes e magnificacao com sinal (paridade)

def A(zeta, s, q):
    z, mu = imagens(zeta, s, q); return np.sum(np.abs(mu)), len(z)

if __name__ == "__main__":
    print("grau do polinomio:", _num.degree())
    for u in [0.1, 0.5, 1.0]:                          # limite q->0 deve dar Paczynski
        print(u, A(u*1j + 1e-9, 1.5, 1e-9)[0], (u*u+2)/(u*np.sqrt(u*u+4)))
    print("fonte longe:", A(0.3+0.05j, 1.2, 1e-3), " perto da caustica central:", A(0.0+0.002j, 1.2, 1e-3))
    for s in [0.6, 1.0, 1.8]:
        zc, ze = caustica(s, 1e-3, 400); print("s =", s, "pontos:", len(ze))

def topologia(s, q, n=3000):
    """Conta curvas criticas separadas (3 = close, 1 = ressonante, 2 = wide)."""
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import connected_components
    zc, _ = caustica(s, q, n)
    P = np.c_[zc.real, zc.imag]; d = cKDTree(P)
    passo = np.median(d.query(P, k=2)[0][:, 1])        # espacamento tipico entre pontos vizinhos
    G = d.sparse_distance_matrix(d, 20 * passo)
    return connected_components(G, directed=False)[0]
