"""Benchmark CNN x busca tradicional com o pyLIMA, versão 2 (tópico 5 da revisão da banca).

Diferenças em relação ao 04_benchmark_pylima.py:
  1. Nenhum evento é descartado. Se uma estratégia falha (exceção do pyLIMA, nenhum ajuste termina), isso conta como FRACASSO
     dela; antes, os eventos em que algo falhava eram retirados da comparação (20 -> 14).
  2. O ponto de partida é trazido para dentro dos limites do pyLIMA (u0 em (−1, 1) etc.); essa era a causa provável das falhas.
  3. Ajuste ORÁCULO, partindo dos parâmetros verdadeiros: dá o χ² de referência (o mínimo "certo"), o que permite separar
     "chegou a um χ² baixo" de "chegou à solução", que era a ressalva do professor.
  4. Busca cega com o MESMO ORÇAMENTO da rede: o mesmo número de ajustes do procedimento em dois ramos (2 ou 3), com partidas
     sorteadas da grade de 8. Isso separa o efeito da INFORMAÇÃO dada pela rede do efeito de simplesmente fazer menos ajustes.
  5. Roda em paralelo (--workers) e pode ser interrompido e retomado: cada evento vira um arquivo JSON em --out.
  6. As estatísticas (taxas de sucesso com IC, teste de McNemar, fator de aceleração com IC) ficam no 14_resumo_benchmark_v2.py.

Estratégias (todas com o mesmo modelo USBL, fonte finita, algoritmo TRF):
  oraculo               1 ajuste partindo dos parâmetros verdadeiros (referência; não é um método utilizável)
  rede_1ramo            1 ajuste partindo da previsão da rede, com o sinal de log s dado por ela
  rede_2ramos           2 ou 3 ajustes partindo de s e 1/s (e de s = 1 se algum candidato for ressonante)
  cega_mesmo_orcamento  o mesmo número de ajustes de rede_2ramos, partidas sorteadas da grade de 8
  cega_8                8 ajustes: s em {0,8; 1,3} × α em {0, π/2, π, 3π/2}, q = 1e-3, (t0, u0, tE) do ajuste PSPL

Tempo: ~3-4 min por evento num processo (o cega_8 domina). Com --workers 4, 100 eventos levam ~1,5-2 h.
Uso:   python scripts/13_benchmark_pylima_v2.py --n 100 --workers 4 --model models/cnn_final.pt
       (interrompeu? rode o MESMO comando de novo: os eventos já feitos são pulados)"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                     # um núcleo por processo; o paralelismo vem de --workers
import sys, io, json, time, pathlib, argparse, contextlib, numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed   # (Pool não serve: o pyLIMA cria processos filhos)
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

GRADE = [(s, al) for s in (0.8, 1.3) for al in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)]
Q_CEGA = 1e-3
A = None                                               # argumentos (preenchidos em cada processo)


def _init(args):
    global A
    A = args
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass


def _ajuste(dados, partida):
    """Um ajuste USBL. Nunca levanta exceção: devolve dict com 'erro' em caso de falha."""
    from mlens.pylima_fit import fit_usbl
    t = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = fit_usbl(dados, partida)
        return {k: (float(v) if k != "limites" else v) for k, v in r.items()}
    except Exception as e:
        return dict(erro=f"{type(e).__name__}: {e}"[:300], seconds=time.time() - t)


def _melhor(ajustes):
    ok = [r for r in ajustes if "erro" not in r]
    out = dict(n_ajustes=len(ajustes), n_falhas=len(ajustes) - len(ok), tempo_s=float(sum(r["seconds"] for r in ajustes)))
    if ok:
        b = min(ok, key=lambda r: r["chi2"])
        out.update({k: b[k] for k in ("chi2", "t0", "u0", "tE", "rho", "s", "q", "alpha")})
    return out


def roda_evento(i):
    arq = pathlib.Path(A.out) / f"evt_{i:05d}.json"
    if arq.exists():
        return i, "já feito"
    from mlens.simulate_pylima import simulate_event_pylima, JD0
    from mlens.features import featurize
    from mlens.pylima_fit import candidatos_s, topologia
    from mlens.torch_cnn import predict
    t_ev = time.time()
    rng = np.random.default_rng(A.seed + i)
    with contextlib.redirect_stdout(io.StringIO()):
        p, data, _ = simulate_event_pylima(rng)
    X, sc, d2, fit = featurize(data)
    t = time.time(); P = predict(A.model, X[None], sc[None], device="cpu"); t_nn = time.time() - t
    dados = {k: (v[0][::A.thin] + JD0, v[1][::A.thin], v[2][::A.thin]) for k, v in data.items()}
    N = int(sum(len(v[0]) for v in dados.values()))
    base_pspl = [JD0 + fit["t0"], fit["u0"], fit["tE"]]
    nn = {k: float(np.asarray(P[k])[0]) for k in ("t0", "u0", "tE", "logq", "abs_logs", "logs", "alpha", "p_wide")}
    base_nn = [JD0 + nn["t0"], nn["u0"], nn["tE"]] if A.base == "rede" else base_pspl
    q_nn = 10 ** nn["logq"]

    E = {}
    E["oraculo"] = [_ajuste(dados, [JD0 + p["t0"], p["u0"], p["tE"], 10 ** p["logs"], 10 ** p["logq"], p["alpha"]])]
    E["rede_1ramo"] = [_ajuste(dados, [*base_nn, 10 ** nn["logs"], q_nn, nn["alpha"]])]
    cand = candidatos_s(nn["abs_logs"], q_nn)
    E["rede_2ramos"] = [dict(_ajuste(dados, [*base_nn, s0, q_nn, nn["alpha"]]), ramo_partida=lab) for lab, s0 in cand]
    E["cega_8"] = [dict(_ajuste(dados, [*base_pspl, s0, Q_CEGA, al]), partida=[s0, al]) for s0, al in GRADE]
    k = len(cand); sel = np.random.default_rng(A.seed + i + 10 ** 6).choice(len(GRADE), size=k, replace=False)
    E["cega_mesmo_orcamento"] = [E["cega_8"][j] for j in sorted(sel)]   # TRF é determinístico: reaproveita os ajustes

    est = {nome: _melhor(aj) for nome, aj in E.items()}
    for nome in est:
        if "s" in est[nome]:
            est[nome]["topologia"] = topologia(est[nome]["s"], est[nome]["q"])
    R = dict(i=i, semente=A.seed + i, N=N, excesso_chi2=float(d2), thin=A.thin, modelo=A.model, base=A.base,
             verdadeiro=dict(t0=p["t0"] + JD0, u0=p["u0"], tE=p["tE"], logq=p["logq"], logs=p["logs"], alpha=p["alpha"],
                             topologia=topologia(10 ** p["logs"], 10 ** p["logq"])),
             pspl=dict(t0=fit["t0"] + JD0, u0=fit["u0"], tE=fit["tE"]),
             rede=dict(nn, t0=nn["t0"] + JD0, tempo_s=t_nn, p_planet=float(np.asarray(P["p_planet"])[0]) if "p_planet" in P else None),
             estrategias=est, ajustes=E, tempo_total_evento_s=time.time() - t_ev)
    tmp = arq.with_suffix(".tmp"); tmp.write_text(json.dumps(R, ensure_ascii=False, default=float), encoding="utf-8"); tmp.replace(arq)
    c = lambda n: est[n].get("chi2", np.nan) / N
    return i, (f"excesso={d2:8.0f}  χ²/N: oráculo={c('oraculo'):.2f} 2ramos={c('rede_2ramos'):.2f} "
               f"cega-mesmo-orç.={c('cega_mesmo_orcamento'):.2f} cega8={c('cega_8'):.2f}  ({time.time() - t_ev:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--thin", type=int, default=2)
    ap.add_argument("--model", default=str(ROOT / "models/cnn_final.pt"))
    ap.add_argument("--base", choices=["rede", "pspl"], default="rede", help="(t0, u0, tE) de partida das estratégias da rede")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out", default=str(ROOT / "data/processed/benchmark_v2"))
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    if not pathlib.Path(a.model).exists():
        sys.exit(f"ERRO: modelo {a.model} não encontrado.")
    ids = [i for i in range(a.start, a.start + a.n) if not (pathlib.Path(a.out) / f"evt_{i:05d}.json").exists()]
    print(f"{a.n - len(ids)} eventos já feitos; faltam {len(ids)}. Saída: {a.out}", flush=True)
    t0 = time.time()
    if a.workers <= 1:
        _init(a); it = map(roda_evento, ids)
    else:
        ex = ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a,))
        it = (f.result() for f in as_completed([ex.submit(roda_evento, i) for i in ids]))
    for k, (i, msg) in enumerate(it, 1):
        el = time.time() - t0
        print(f"[{k}/{len(ids)}] evento {i:4d}  {msg}   | decorrido {el / 60:.0f} min, restam ~{el / k * (len(ids) - k) / 60:.0f} min", flush=True)
    print("pronto. Agora rode:  python scripts/14_resumo_benchmark_v2.py")
