"""CNN 1D (PyTorch) sobre as features de resíduo. Treinar na SUA máquina (GPU ajuda, CPU funciona).
   Não testado neste ambiente (sem PyTorch) — qualquer erro de forma/shape aparece já na 1ª época."""
import numpy as np, torch, torch.nn as nn


class Net(nn.Module):
    def __init__(self, cin=3, nsc=5, nout=8, nb=256):
        super().__init__()
        ch, L, c = [32, 64, 128, 128], [], cin
        for o in ch:
            L += [nn.Conv1d(c, o, 7, padding=3), nn.BatchNorm1d(o), nn.GELU(), nn.MaxPool1d(2)]; c = o
        self.conv = nn.Sequential(*L); nflat = c * (nb // 2 ** len(ch))
        self.head = nn.Sequential(nn.Linear(nflat + nsc, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, nout))

    def forward(self, x, sc):
        return self.head(torch.cat([self.conv(x).flatten(1), sc], 1))


def prep_x(X):
    X = np.array(X, np.float32, copy=True); X[:, 1] /= 5.0; return X          # z-score do resíduo -> ~[-4, 4]


def make_targets(y):
    """y (N,7)=[t0,u0,tE,logq,logs,cosα,sinα] -> reg (N,7) com |logs| (degenerescência s<->1/s) e flag (s>1)."""
    reg = np.stack([y[:, 0], y[:, 1], y[:, 2], y[:, 3], np.abs(y[:, 4]), y[:, 5], y[:, 6]], 1).astype(np.float32)
    return reg, (y[:, 4] > 0).astype(np.float32)


def decode(out, ymean, ystd):
    """saída da rede (N,8) -> dict de parâmetros físicos."""
    reg = out[:, :7] * ystd + ymean; p_wide = 1 / (1 + np.exp(-out[:, 7]))
    a = np.arctan2(reg[:, 6], reg[:, 5]) % (2 * np.pi); sgn = np.where(p_wide > 0.5, 1.0, -1.0)
    return dict(t0=reg[:, 0], u0=reg[:, 1], tE=reg[:, 2], logq=reg[:, 3], abs_logs=reg[:, 4], logs=sgn * reg[:, 4],
                alpha=a, p_wide=p_wide)


@torch.no_grad()
def predict(path, X, sc, device=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=dev, weights_only=False); net = Net().to(dev); net.load_state_dict(ck["state"]); net.eval()
    x = torch.tensor(prep_x(X)).to(dev); s = torch.tensor((sc - ck["scm"]) / ck["scs"], dtype=torch.float32).to(dev)
    return decode(net(x, s).cpu().numpy(), ck["ym"], ck["ys"])
