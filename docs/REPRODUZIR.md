# Reprodução dos resultados da monografia

**Versão de referência:** os resultados da monografia correspondem à tag `v1.0-tcc` deste repositório.
O commit exato, as sementes, os comandos executados e o hash SHA-256 de cada modelo e conjunto de dados ficam em
`models/MANIFESTO_final.json`, gerado automaticamente pelo `reproduzir.py`. Todos os arquivos da versão final têm o
sufixo `final` (ex.: `models/cnn_final.pt`, `models/r2_final.json`). Arquivos de versões anteriores (v1, v2a, v2b, v3)
estão em `models/historico/` e **não** correspondem ao texto.

## Como rodar

```bash
pip install -r requirements.txt                 # + PyTorch: pip install torch
python reproduzir.py --listar                   # mostra todos os comandos, sem rodar
python reproduzir.py --rapido                   # teste de fumaça (~5 min): só confere se tudo funciona
python reproduzir.py                            # dados + 5 treinos + análises (~4 h em CPU)
python reproduzir.py --etapas benchmark         # benchmark com o pyLIMA (~2 h com --workers 4)
python reproduzir.py --etapas ogle              # eventos reais do OGLE (precisa de internet)
```

Cada passo é pulado se a saída já existir; se algo falhar, corrija e rode o mesmo comando de novo.
`--forcar` refaz tudo. Os conjuntos de dados (`data/`) não são versionados, mas são regenerados de forma idêntica
pelas sementes abaixo.

## Sementes

Cada conjunto usa as sementes `s, s+1, …, s+n−1`, uma por evento, sem sobreposição entre conjuntos.

| Conjunto | Arquivo | n | Semente inicial | Observação |
|---|---|---|---|---|
| Treino (regressão) | `train_mix.npz` | 30 000 | 42 | 70% das trajetórias ancoradas na cáustica; 70/10/20 treino/validação/teste-mix |
| Treino (detecção) | `train_null.npz` | 30 000 | 100 000 | 20% sem planeta |
| Teste uniforme | `test_uniform.npz` | 6 000 | 900 000 | priori uniforme, sem ancoragem |
| Sem planeta (calibração) | `test_null.npz` | 3 000 | 910 000 | escolhe o limiar de 5% de falsos positivos |
| Sem planeta (controle) | `test_null_vbm.npz` | 3 000 | 920 000 | VBMicrolensing com q = 1e-9, s = 10 |
| Sem planeta (independente) | `test_null_indep.npz` | 3 000 | 930 000 | avaliação dos eventos 1L1S (Seção 4.7) |
| Benchmark pyLIMA | — | 100 | 2 026 | eventos simulados com o pyLIMA, com luz de mistura |

Treinos: semente 0 (e 1 para a segunda semente), 30 épocas, AdamW, taxa máxima 2e-3, lotes de 128.

## O que gera o quê

| Na monografia | Script | Saída |
|---|---|---|
| Tabela 3 (erros) | `02b_treinar_cnn.py --run final` | `models/metrics_cnn_final.json` (chave `uniform`) |
| Tabela 4, Figura 7 (R², previsto × verdadeiro) | `07_r2.py` | `models/r2_final.json`, `figures/fig_r2_final.png` |
| **Nova tabela: rede × ajuste PSPL** (tópico 2) | `12_rede_vs_pspl.py --run final` | `models/rede_vs_pspl_final.json`, `figures/fig_rede_vs_pspl_final.png` |
| Figura 8 (erro × excesso de χ²) | `06_analise_s.py --run final` | `figures/fig_erro_vs_excesso_final.png` |
| Figura 9 (sinal de log s) | `06_analise_s.py --run final` | `figures/fig_s_analise_final.png`, `models/analise_s_final.json` |
| Tabela 5 (ablação) | `07_r2.py` nos runs `abl_sem_u`, `abl_u`, `final` | `models/r2_abl_sem_u.json`, `models/r2_abl_u.json`, `models/r2_final.json` |
| Tabela 12 (duas sementes) | `07_r2.py` nos runs `final`, `final_s1` | `models/r2_final.json`, `models/r2_final_s1.json` |
| Regime planetário, q < 0,03 (tópico 3) | `15_regime_planetario.py --run final` | `models/regime_planetario_final.json` |
| Tabela 6, Figura 10 (detecção) | `10_deteccao.py --run final_det` | `models/deteccao_final_det.json`, `figures/fig_deteccao_final_det.png` |
| Detecção só no regime planetário, q < 0,03 (tópico 3) | `10_deteccao.py --logq-max -1.52` | `models/deteccao_final_det_logq-1.52.json` |
| Controle da detecção | `10_deteccao.py --null-file test_null_vbm.npz` | `models/deteccao_final_det_test_null_vbm.json` |
| Tabela 7, Figura 11 (eventos 1L1S) | `11_avaliar_1l1s.py --run final_det` | `models/avaliacao_1l1s_final_det.json`, `figures/fig_1l1s_final_det.png` |
| Tabela 8, Figura 12 (pyLIMA) | `13_benchmark_pylima_v2.py` + `14_resumo_benchmark_v2.py` | `models/benchmark_v2_resumo.json`, `figures/fig_benchmark_v2.png` |
| Tabela 9, Figura 13 (OGLE) | `05_ogle.py --event 2012-BLG-0406` (e 2016-BLG-1195) | terminal, `figures/ogle_*.png` |
| Figura 3 (topologias) | `03b_figuras_cnn.py` | `figures/fig_topologias.png` |
| Figura 4 (degenerescência) | `08_degenerescencia_s.py` | `figures/fig_degenerescencia_s.png` |
| Figura 6 (evento exemplo) | `03b_figuras_cnn.py` | `figures/fig_evento.png` |

## Mudanças da versão final em relação às anteriores

1. **t0, u0, tE como correção ao ajuste PSPL** (`--delta`). A rede prevê δt0 = (t0 − t0_PSPL)/tE_PSPL,
   δu0 = log10(u0/u0_PSPL) e δtE = log10(tE/tE_PSPL), comprimidos por asinh. Saída zero = repetir o ajuste.
   A referência é o ajuste PSPL limitado a uma faixa física, porque em ~8% dos eventos o ajuste diverge.
2. **Comparação direta com o ajuste PSPL** (`12_rede_vs_pspl.py`): erro dos dois estimadores no mesmo evento,
   teste de Wilcoxon pareado e a fração da correção aplicada, β.
3. **Eventos 1L1S**: o limiar é escolhido em `test_null.npz` e a avaliação é feita em `test_null_indep.npz`.
4. **Benchmark pyLIMA v2**: nenhum evento descartado; ajuste oráculo como referência; busca cega com o mesmo
   orçamento da rede; IC de 95%, teste de McNemar e fator de aceleração por custo por sucesso.
5. **Proveniência**: cada treino grava commit, semente, argumentos e versões em `models/meta_cnn_<run>.json`.

## Autoria do código e uso de inteligência artificial

O código foi desenvolvido por Danilo Parmejane Menezes, com auxílio do assistente de IA Claude (Anthropic) na escrita,
na depuração e na revisão de partes do código. As bibliotecas usadas são VBMicrolensing (Bozza), pyLIMA (Bachelet et al.)
e PyTorch.
