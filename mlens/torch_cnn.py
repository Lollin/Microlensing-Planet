"""CNN 1D (PyTorch) sobre as features de resíduo. Treinar na SUA máquina (GPU ajuda, CPU funciona).

Opções da versão 2 (gravadas no checkpoint em ck["cfg"]; checkpoints antigos continuam funcionando):
- uchan: acrescenta um 4º canal com log10 u(τ), u = sqrt(u0² + τ²), usando o u0 do ajuste 1L1S.
  Motivação física: a fonte cruza a cáustica planetária quando u(t) ≈ |s − 1/s|.
- target: "abslogs" (alvo |log s|, versão 1) ou "x" (alvo |s − 1/s|, linear na posição da cáustica planetária).
- null: 9ª saída = logit da probabilidade de haver planeta (treino com eventos de lente simples)."""
import numpy as np, torch, torch.nn as nn

NB, TAU = 256, 4.0                                             # iguais a mlens/features.py
TAU_C = (np.linspace(-TAU, TAU, NB + 1)[1:] + np.linspace(-TAU, TAU, NB + 1)[:-1]) / 2


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


def prep_x(X, sc=None, uchan=False):
    """X (N,3,NB) -> entrada da rede. sc são os escalares CRUS (antes da normalização): sc[:,1] = log10 u0 do ajuste 1L1S."""
    X = np.array(X, np.float32, copy=True); X[:, 1] /= 5.0           # z-score do resíduo -> ~[-4, 4]
    if uchan:
        u0 = 10.0 ** np.asarray(sc, np.float64)[:, 1]
        lu = np.log10(np.sqrt(u0[:, None] ** 2 + TAU_C[None, :] ** 2))   # entre ~-2 e ~0,6
        X = np.concatenate([X, lu[:, None, :].astype(np.float32)], 1)
    return X


def make_targets(y, target="abslogs"):
    """y (N,7)=[t0,u0,tE,logq,logs,cosα,sinα] -> reg (N,7) e flag (s>1).
    Coluna 4: |log s| (target="abslogs") ou |s − 1/s| (target="x"). Ambos são invariantes por s -> 1/s."""
    ls = np.abs(y[:, 4])
    sep = ls if target == "abslogs" else (10 ** ls - 10 ** -ls)
    reg = np.stack([y[:, 0], y[:, 1], y[:, 2], y[:, 3], sep, y[:, 5], y[:, 6]], 1).astype(np.float32)
    return reg, (y[:, 4] > 0).astype(np.float32)


def decode(out, ymean, ystd, target="abslogs"):
    """saída da rede (N,8 ou 9) -> dict de parâmetros físicos (+ p_planet se a rede tiver a 9ª saída)."""
    P = _decode8(out, ymean, ystd, target)
    if out.shape[1] > 8:
        P["p_planet"] = 1 / (1 + np.exp(-out[:, 8]))
    return P


def _decode8(out, ymean, ystd, target):
    """saída da rede (N,8) -> dict de parâmetros físicos."""
    reg = out[:, :7] * ystd + ymean; p_wide = 1 / (1 + np.exp(-out[:, 7]))
    if target == "abslogs":
        abs_logs = np.clip(reg[:, 4], 0, None)
    else:                                                          # x = s − 1/s, s > 1  ->  s = (x + sqrt(x² + 4)) / 2
        x = np.clip(reg[:, 4], 0, None); abs_logs = np.log10((x + np.sqrt(x ** 2 + 4)) / 2)
    a = np.arctan2(reg[:, 6], reg[:, 5]) % (2 * np.pi); sgn = np.where(p_wide > 0.5, 1.0, -1.0)
    return dict(t0=reg[:, 0], u0=reg[:, 1], tE=reg[:, 2], logq=reg[:, 3], abs_logs=abs_logs, logs=sgn * abs_logs,
                alpha=a, p_wide=p_wide)


@torch.no_grad()
def predict(path, X, sc, device=None, chunk=1000):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=dev, weights_only=False); cfg = ck.get("cfg", {})
    uchan, target = cfg.get("uchan", False), cfg.get("target", "abslogs")
    net = Net(cin=3 + int(uchan), nsc=len(ck["scm"]), nout=9 if cfg.get("null") else 8).to(dev); net.load_state_dict(ck["state"]); net.eval()
    outs = []
    for i in range(0, len(X), chunk):
        x = torch.tensor(prep_x(X[i:i + chunk], sc[i:i + chunk], uchan)).to(dev)
        s = torch.tensor((sc[i:i + chunk] - ck["scm"]) / ck["scs"], dtype=torch.float32).to(dev)
        outs.append(net(x, s).cpu().numpy())
    return decode(np.concatenate(outs), ck["ym"], ck["ys"], target)
