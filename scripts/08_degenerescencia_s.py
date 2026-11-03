"""Figura da degenerescência close-wide (s <-> 1/s) para o TCC. Não precisa de dataset: calcula tudo pela equação da lente.
Uso (da raiz do repositório):  python scripts/08_degenerescencia_s.py
Saída: figures/fig_degenerescencia_s.png
Depende de scripts/binaria.py (mesma pasta) e do pacote sympy."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from binaria import caustica, posicoes, A
q, sw = 1e-3, 1.5; sc = 1/sw
cores = {"close": "#2c6fbb", "wide": "#d1495b"}
def central(s):
    m1, m2, a, b = posicoes(s, q); _, ze = caustica(s, q, 4000)
    host = a; c = ze[np.abs(ze - host) < 0.02]; return ze, c, (c.real.min()+c.real.max())/2 + 0j
dados = {"close": central(sc), "wide": central(sw)}
for k, (_, c, cen) in dados.items():
    print(k, "meia-largura em x:", np.ptp((c - cen).real)/2, " em y:", np.ptp((c - cen).imag)/2)
print("previsto (Chung et al. 2005), meia-largura 2q/(s-1/s)^2 =", 2*q/(sw-1/sw)**2)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for k, s in [("close", sc), ("wide", sw)]:
    ze, c, cen = dados[k]; m1, m2, a, b = posicoes(s, q)
    ax[0].plot(ze.real - a, ze.imag, ".", ms=1.5, color=cores[k], label=f"{k}: s = {s:.2f}")
    ax[1].plot((c - cen).real*1e3, (c - cen).imag*1e3, ".", ms=1.2, color=cores[k], label=f"{k}: s = {s:.2f}")
ax[0].plot(0, 0, "k*", ms=12); ax[0].set_ylim(-0.12, 0.12); ax[0].legend(markerscale=12, fontsize=9, loc="lower center")
ax[0].annotate("cáusticas\nplanetárias (close)", (-0.83, 0.075), (-0.78, 0.09), fontsize=8, color=cores["close"])
ax[0].annotate("cáustica\nplanetária (wide)", (0.84, 0.0), (0.45, -0.1), fontsize=8, color=cores["wide"], arrowprops=dict(arrowstyle="->", color=cores["wide"]))
ax[0].set_title("(a) Cáusticas completas (estrela em 0)"); ax[0].set_xlabel("x / θE"); ax[0].set_ylabel("y / θE")
ax[1].set_aspect("equal"); ax[1].legend(markerscale=8, fontsize=9)
ax[1].set_title("(b) Cáusticas centrais sobrepostas"); ax[1].set_xlabel("x (10⁻³ θE)"); ax[1].set_ylabel("y (10⁻³ θE)")
u0, al, tE = 0.0012, np.radians(60), 30.0
t = np.linspace(-0.012, 0.012, 1500)*tE
for k, s in [("close", sc), ("wide", sw)]:
    cen = dados[k][2]; tau = t/tE
    zeta = cen + (tau*np.cos(al) - u0*np.sin(al)) + 1j*(tau*np.sin(al) + u0*np.cos(al))
    Am = np.array([A(z, s, q)[0] for z in zeta]); dados[k] = dados[k] + (Am,)
    ax[2].plot(t*24, Am, color=cores[k], lw=2 if k == "close" else 1.2, ls="-" if k == "close" else "--", label=f"{k}: s = {s:.2f}")
u = np.sqrt(u0**2 + (t/tE)**2); ax[2].plot(t*24, (u*u+2)/(u*np.sqrt(u*u+4)), color="gray", lw=1, ls=":", label="sem planeta (PSPL)")
ax[2].set_xlabel("t − t₀ (horas)"); ax[2].set_ylabel("magnificação A"); ax[2].legend(fontsize=9)
ax[2].set_title("(c) Pico da curva de luz (u₀ = 0,0012, α = 60°)")
plt.tight_layout(); (ROOT / "figures").mkdir(exist_ok=True); plt.savefig(ROOT / "figures/fig_degenerescencia_s.png", dpi=200); print("salvo: figures/fig_degenerescencia_s.png")
