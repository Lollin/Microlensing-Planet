"""Avalia a CNN já treinada em OUTRO conjunto (ex.: simulado com pyLIMA) — teste de generalização entre simuladores.
Uso:  python scripts/02d_avaliar_cnn.py --data data/processed/feat_pylima.npz --model models/cnn_2l1s.pt"""
import sys, pathlib, json, argparse, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mlens.torch_cnn import predict

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/processed/feat_pylima.npz"); ap.add_argument("--model", default="models/cnn_2l1s.pt")
ap.add_argument("--out", default="models/metrics_cnn_pylima.json"); a = ap.parse_args()
from mlens import metrics as M
d = np.load(a.data); P = predict(a.model, d["X"], d["sc"]); rep = M.full_report(d["y"], P, d["dchi2"]); M.print_report(rep, "teste em " + a.data)
json.dump({"pylima": rep}, open(a.out, "w"), indent=2)
