# Fluxo do banner (só CNN 1D)

```
python -m venv .venv && (Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate)
pip install -r requirements-extra.txt
python -c "import torch, VBMicrolensing, reportlab; print('ok', torch.__version__)"

python scripts/01b_gerar_dataset_feat.py --n 20000 --workers 4 --caustic-frac 0.7
python scripts/02b_treinar_cnn.py --epochs 60 --min-dchi2 100
python scripts/03b_figuras_cnn.py
python banner/build_banner.py          # -> banner/banner_SAB.pdf
```
Depois (não necessário para o banner): `04_benchmark_pylima.py`, `05_ogle.py`.

## pyLIMA (comparativo; opcional para o banner de hoje)
O pyLIMA usa a mesma convenção de (t0, u0, tE, s, q, α) do VBMicrolensing (verificado: diferença relativa < 2e-4).
```
python scripts/01b_gerar_dataset_feat.py --n 3000 --engine pylima --workers 4 --out data/processed/feat_pylima.npz   # ~10-15 min (1 núcleo)
python scripts/02d_avaliar_cnn.py          # avalia a CNN (treinada em VBM) nesse conjunto -> models/metrics_cnn_pylima.json
python banner/build_banner.py              # a tabela ganha sozinha a coluna "2L1S (pyLIMA)"
```
