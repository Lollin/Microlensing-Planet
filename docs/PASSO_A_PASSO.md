# Fluxo (CNN 1D, avaliação com métricas de erro)

Sementes diferentes entre treino e teste, para não haver eventos repetidos (o gerador usa as sementes seed..seed+n-1).

```
python scripts/01b_gerar_dataset_feat.py --n 30000 --workers 4 --caustic-frac 0.7 --seed 42     --out data/processed/train_mix.npz
python scripts/01b_gerar_dataset_feat.py --n 6000  --workers 4 --caustic-frac 0   --seed 900000 --out data/processed/test_uniform.npz
python scripts/02b_treinar_cnn.py --epochs 30
python scripts/03b_figuras_cnn.py
python banner/build_banner.py        # feche o PDF antigo antes
```

- `train_mix`: 70% das trajetórias ancoradas em cáusticas (só para o treino).
- `test_uniform`: priori uniforme, sem ancoragem. É a população usada nos resultados do banner.
- O treino usa todos os eventos (`--min-excess 0`). Para o experimento antigo, `--min-excess 100`.
- Curva de aprendizado: rode `02b` com `--n-train 5000`, `10000`, `20000` (mude `--out` para não sobrescrever) e compare o erro de teste.
- Todos os números do banner vêm de `models/metrics_cnn.json` e `models/meta_cnn.json`; nada é digitado à mão.

## pyLIMA (comparativo, depois)
```
pip install -r requirements-pylima.txt
python scripts/01b_gerar_dataset_feat.py --n 3000 --engine pylima --caustic-frac 0 --seed 800000 --out data/processed/feat_pylima.npz
python scripts/02d_avaliar_cnn.py
```
