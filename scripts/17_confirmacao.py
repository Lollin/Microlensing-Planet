"""Conjunto de CONFIRMAÇÃO: avalia a rede final em um conjunto de teste com semente nova, uma única vez.

Motivo: as decisões de projeto (prever correções ao PSPL, canal u(τ), alvo |s − 1/s|) foram tomadas olhando o teste uniforme.
Esse teste deixou, portanto, de ser totalmente independente. Este conjunto nunca foi olhado durante o desenvolvimento, e as
mesmas métricas são repetidas nele. Se os números forem compatíveis com os do teste uniforme, o teste uniforme não foi
sobreajustado pelas escolhas de projeto.
Uso:   python scripts/17_confirmacao.py --run final --test data/processed/test_confirmacao.npz
Saída: models/confirmacao_<run>.json  (métricas da rede, R² e comparação com o PSPL, iguais às do teste uniforme)"""
import sys, json, pathlib, argparse, subprocess, numpy as np
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mlens.torch_cnn import predict
from mlens import metrics as M

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="final"); ap.add_argument("--test", default=str(ROOT / "data/processed/test_confirmacao.npz"))
a = ap.parse_args()
d = np.load(a.test); P = predict(str(ROOT / f"models/cnn_{a.run}.pt"), d["X"], d["sc"])
rep = M.full_report(d["y"], P, d["dchi2"]); M.print_report(rep, f"CONFIRMAÇÃO ({a.test})")
pred = ROOT / f"data/processed/test_pred_confirmacao_{a.run}.npz"
np.savez(pred, y=d["y"], d2=d["dchi2"], sc=d["sc"], **{k: np.asarray(v) for k, v in P.items()})
# mesmas análises do teste uniforme, com tag própria
subprocess.run([sys.executable, str(ROOT / "scripts/12_rede_vs_pspl.py"), "--pred", str(pred), "--test", a.test, "--tag", f"confirmacao_{a.run}"], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts/07_r2.py"), "--pred", str(pred), "--tag", f"confirmacao_{a.run}"], check=True)
json.dump({"uniform": rep, "arquivo": a.test, "n": int(len(d["y"]))}, open(ROOT / f"models/confirmacao_{a.run}.json", "w", encoding="utf-8"), indent=1)
print(f"salvo: models/confirmacao_{a.run}.json, models/rede_vs_pspl_confirmacao_{a.run}.json, models/r2_confirmacao_{a.run}.json")
