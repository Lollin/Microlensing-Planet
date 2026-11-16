# Roteiro: como rodar e como enviar ao Overleaf

Este é o único arquivo que você precisa seguir. Tudo o que está abaixo roda a partir da pasta do projeto.

## 1. Rodar tudo (uma vez, ~6 h, pode deixar sozinho)

```bash
python reproduzir.py --rapido                 # 5 min: confere se tudo funciona
python reproduzir.py --benchmark --ogle       # a rodada de verdade
```

Se parar no meio (energia, erro), rode o mesmo comando de novo: ele continua de onde parou.
No fim aparece `manifesto: models/MANIFESTO_final.json`.

## 2. Conferir

```bash
python reproduzir.py --etapas latex           # regenera latex_tcc/ (já roda no fim do passo 1)
```

Abra `latex_tcc/resultados/numeros.tex`. Se alguma linha tiver `\pendente{...}`, algum resultado faltou;
o terminal diz qual arquivo não foi encontrado.

## 3. Enviar ao Overleaf

No Overleaf, apague as pastas `resultados/` e `figuras/` do projeto e envie no lugar delas as pastas
`latex_tcc/resultados/` e `latex_tcc/figuras/`. Recompile 3 ou 4 vezes (plano gratuito).
Os números do texto, as tabelas e as figuras se atualizam sozinhos.

## 4. Fechar a versão da monografia

```bash
git add -A
git commit -m "Resultados da versão final da monografia"
git tag -a v1.0-tcc -m "Versão enviada à banca"
git push && git push --tags
```

Só depois disso faça o merge da branch `revisao-banca` na principal (pelo site do GitHub ou com
`git checkout main && git merge revisao-banca && git push`).

## Se der erro

Copie a mensagem inteira do terminal, da linha `[roda] python ...` até o fim, e envie.
