# Microlensing-Planet

Inferência de parâmetros de eventos de microlente gravitacional **2L1S** (lente binária / planetária) com redes neurais, a partir de curvas de luz sintéticas realistas (lacunas, ruído gaussiano, vários telescópios).

## Estrutura
```
mlens/        código-fonte (config, simulate, dataset, model, plots)
scripts/      01_gerar_dataset.py -> 02_treinar.py -> 03_figuras.py
data/         raw/ e processed/ (NÃO versionados, ver .gitignore)
models/       modelos treinados e metrics.json
figures/      figuras finais (entram no banner)
banner/       texto e arquivos do banner
docs/         PLANO.md e notas
notebooks/    exploração (1L1S antigo entra em notebooks/legacy_1L1S/)
```

## Rodar (≈2 min em CPU)
```bash
pip install -r requirements.txt
python scripts/01_gerar_dataset.py --n 6000
python scripts/02_treinar.py
python scripts/03_figuras.py
```
Tudo que muda o experimento (prioris, telescópios, lacunas, ruído) fica em `mlens/config.py`.

## Resultado atual (baseline)
Ver `models/metrics.json`. A MLP recupera t0, u0, tE; **q, s, α ainda não** (próximos passos em `docs/PLANO.md`).
