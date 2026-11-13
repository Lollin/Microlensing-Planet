"""CNN 1D (PyTorch) sobre as features de resíduo. Treinar na SUA máquina (GPU ajuda, CPU funciona).

Opções (gravadas no checkpoint em ck["cfg"]; checkpoints antigos continuam funcionando):
- uchan: acrescenta um 4º canal com log10 u(τ), u = sqrt(u0² + τ²), usando o u0 do ajuste 1L1S.
  Motivação física: a fonte cruza a cáustica planetária quando u(t) ≈ |s − 1/s|.
- target: "abslogs" (alvo |log s|, versão 1) ou "x" (alvo |s − 1/s|, linear na posição da cáustica planetária).
- null: 9ª saída = logit da probabilidade de haver planeta (treino com eventos de lente simples).
- delta (versão final): t0, u0 e tE deixam de ser previstos em valor absoluto e passam a ser previstos como
  CORREÇÃO ao ajuste de lente simples (PSPL):
      δt0 = (t0 − t0_PSPL) / tE_PSPL        (em unidades de tE: a mesma escala da grade em τ)
      δu0 = log10(u0 / u0_PSPL)
      δtE = log10(tE / tE_PSPL)
  Os alvos entram comprimidos, y = asinh(δ / DELTA_ESCALA): linear para correções pequenas (a maioria, ~0,002 tE em t0)
  e logarítmico para as grandes; sem isso o desvio-padrão do alvo é dominado pelos poucos eventos com correção grande e a
  rede não resolve as correções típicas.
  Saída zero = "repete o ajuste PSPL". Assim a pergunta "a rede corrige ou só reproduz o ajuste?" vira a pergunta
  "a correção prevista é diferente de zero e vai na direção certa?", que o script 12_rede_vs_pspl.py mede."""
import numpy as np, torch, torch.nn as nn
from . import config as C

NB, TAU = 256, 4.0                                             # iguais a mlens/features.py
TAU_C = (np.linspace(-TAU, TAU, NB + 1)[1:] + np.linspace(-TAU, TAU, NB + 1)[:-1]) / 2
DELTA_CLIP = (2.0, 1.5, 1.5)                                   # limites dos alvos δt0 (tE), δu0 (dex), δtE (dex) no treino
DELTA_ESCALA = (0.002, 0.02, 0.02)                             # escala da compressão asinh (≈ correção típica de cada um)
# Faixa física da REFERÊNCIA das correções. Em ~8% dos eventos de teste (mais nos de anomalia forte) o ajuste PSPL diverge
# (ex.: tE ~ 1e9 d, u0 ~ 1e-9: mistura enorme). Sem limitar, a correção relativa a ele não tem significado.
REF_T0 = (C.T_MIN, C.T_MAX); REF_U0 = (1e-3, 3.0); REF_TE = (1.0, 1000.0)
# Faixa dos escalares de ENTRADA (só nas redes novas, cfg["clip_sc"]): log10 tE, log10 u0, fs, t0 normalizado.
SC_CLIP = ((0.0, 3.0), (-3.0, 0.5), (-1.0, 5.0), (0.0, 1.0))


class Net(nn.Module):
    def __init__(self, cin=3, nsc=4, nout=8, nb=256, drop=0.2):
        super().__init__()
        ch, L, c = [32, 64, 128, 128], [], cin
        for o in ch:
            L += [nn.Conv1d(c, o, 7, padding=3), nn.BatchNorm1d(o), nn.GELU(), nn.MaxPool1d(2)]; c = o
        self.conv = nn.Sequential(*L); nflat = c * (nb // 2 ** len(ch))
        self.head = nn.Sequential(nn.Linear(nflat + nsc, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, nout))

    def forward(self, x, sc):
        return self.head(torch.cat([self.conv(x).flatten(1), sc], 1))


def pspl_from_sc(sc):
    """Escalares CRUS sc = [log10 tE, log10 u0, fs, t0 normalizado] -> (t0, u0, tE) do ajuste de lente simples."""
    sc = np.asarray(sc, np.float64)
    return C.T_MIN + sc[:, 3] * (C.T_MAX - C.T_MIN), 10.0 ** sc[:, 1], 10.0 ** sc[:, 0]


def pspl_fisico(sc):
    """True onde o ajuste PSPL caiu dentro da faixa física (REF_*)."""
    t0, u0, tE = pspl_from_sc(sc)
    return (t0 >= REF_T0[0]) & (t0 <= REF_T0[1]) & (u0 >= REF_U0[0]) & (u0 <= REF_U0[1]) & (tE >= REF_TE[0]) & (tE <= REF_TE[1])


def ref_from_sc(sc):
    """Referência das correções: o ajuste PSPL limitado à faixa física."""
    t0, u0, tE = pspl_from_sc(sc)
    return np.clip(t0, *REF_T0), np.clip(u0, *REF_U0), np.clip(tE, *REF_TE)


def clip_sc(sc):
    """Escalares de entrada limitados (redes novas). Evita que um ajuste PSPL divergente (log tE = 9) domine a normalização."""
    sc = np.array(sc, np.float32, copy=True)
    for j, (lo, hi) in enumerate(SC_CLIP):
        sc[:, j] = np.clip(sc[:, j], lo, hi)
    return sc


def prep_x(X, sc=None, uchan=False):
    """X (N,3,NB) -> entrada da rede. sc são os escalares CRUS (antes da normalização): sc[:,1] = log10 u0 do ajuste 1L1S."""
    X = np.array(X, np.float32, copy=True); X[:, 1] /= 5.0           # z-score do resíduo -> ~[-4, 4]
    if uchan:
        u0 = 10.0 ** np.asarray(sc, np.float64)[:, 1]
        lu = np.log10(np.sqrt(u0[:, None] ** 2 + TAU_C[None, :] ** 2))   # entre ~-2 e ~0,6
        X = np.concatenate([X, lu[:, None, :].astype(np.float32)], 1)
    return X


def make_targets(y, target="abslogs", sc=None, delta=False):
    """y (N,7)=[t0,u0,tE,logq,logs,cosα,sinα] -> reg (N,7) e flag (s>1).
    Coluna 4: |log s| (target="abslogs") ou |s − 1/s| (target="x"). Ambos são invariantes por s -> 1/s.
    delta=True: colunas 0-2 viram a correção ao ajuste PSPL (ver docstring do módulo); exige sc."""
    ls = np.abs(y[:, 4])
    sep = ls if target == "abslogs" else (10 ** ls - 10 ** -ls)
    c0, c1, c2 = y[:, 0], y[:, 1], y[:, 2]
    if delta:
        t0f, u0f, tEf = ref_from_sc(sc)
        c0 = np.clip((y[:, 0] - t0f) / tEf, -DELTA_CLIP[0], DELTA_CLIP[0])
        c1 = np.clip(np.log10(np.clip(y[:, 1], 1e-4, None) / u0f), -DELTA_CLIP[1], DELTA_CLIP[1])
        c2 = np.clip(np.log10(y[:, 2] / tEf), -DELTA_CLIP[2], DELTA_CLIP[2])
    if delta:
        c0, c1, c2 = (np.arcsinh(c / e) for c, e in zip((c0, c1, c2), DELTA_ESCALA))
    reg = np.stack([c0, c1, c2, y[:, 3], sep, y[:, 5], y[:, 6]], 1).astype(np.float32)
    return reg, (y[:, 4] > 0).astype(np.float32)


def decode(out, ymean, ystd, target="abslogs", sc=None, delta=False):
    """saída da rede (N,8 ou 9) -> dict de parâmetros físicos (+ p_planet se a rede tiver a 9ª saída)."""
    P = _decode8(out, ymean, ystd, target, sc, delta)
    if out.shape[1] > 8:
        P["p_planet"] = 1 / (1 + np.exp(-out[:, 8]))
    return P


def _decode8(out, ymean, ystd, target, sc=None, delta=False):
    """saída da rede (N,8) -> dict de parâmetros físicos."""
    reg = out[:, :7] * ystd + ymean; p_wide = 1 / (1 + np.exp(-out[:, 7]))
    if target == "abslogs":
        abs_logs = np.clip(reg[:, 4], 0, None)
    else:                                                          # x = s − 1/s, s > 1  ->  s = (x + sqrt(x² + 4)) / 2
        x = np.clip(reg[:, 4], 0, None); abs_logs = np.log10((x + np.sqrt(x ** 2 + 4)) / 2)
    a = np.arctan2(reg[:, 6], reg[:, 5]) % (2 * np.pi); sgn = np.where(p_wide > 0.5, 1.0, -1.0)
    if delta:
        if sc is None:
            raise ValueError("rede treinada com delta=True: decode precisa dos escalares sc do ajuste PSPL")
        t0f, u0f, tEf = ref_from_sc(sc)                         # correções limitadas à faixa vista no treino (sem extrapolação)
        d0, d1, d2 = (np.clip(DELTA_ESCALA[j] * np.sinh(np.clip(reg[:, j], -30, 30)), -DELTA_CLIP[j], DELTA_CLIP[j]) for j in range(3))
        t0, u0, tE = t0f + d0 * tEf, u0f * 10.0 ** d1, tEf * 10.0 ** d2
    else:
        t0, u0, tE = reg[:, 0], reg[:, 1], reg[:, 2]
    return dict(t0=t0, u0=u0, tE=tE, logq=reg[:, 3], abs_logs=abs_logs, logs=sgn * abs_logs,
                alpha=a, p_wide=p_wide)


@torch.no_grad()
def predict(path, X, sc, device=None, chunk=1000):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=dev, weights_only=False); cfg = ck.get("cfg", {})
    uchan, target, delta = cfg.get("uchan", False), cfg.get("target", "abslogs"), cfg.get("delta", False)
    sc_raw = np.asarray(sc); sc = clip_sc(sc_raw) if cfg.get("clip_sc") else sc_raw
    net = Net(cin=3 + int(uchan), nsc=len(ck["scm"]), nout=9 if cfg.get("null") else 8).to(dev); net.load_state_dict(ck["state"]); net.eval()
    outs = []
    for i in range(0, len(X), chunk):
        x = torch.tensor(prep_x(X[i:i + chunk], sc[i:i + chunk], uchan)).to(dev)
        s = torch.tensor((sc[i:i + chunk] - ck["scm"]) / ck["scs"], dtype=torch.float32).to(dev)
        outs.append(net(x, s).cpu().numpy())
    return decode(np.concatenate(outs), ck["ym"], ck["ys"], target, sc_raw, delta)
