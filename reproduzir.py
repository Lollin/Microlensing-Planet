"""Reproduz TODOS os resultados da versão final da monografia, com sementes fixas.

    python reproduzir.py                      # dados + treinos + análises (sem benchmark e sem OGLE)
    python reproduzir.py --benchmark          # + benchmark com o pyLIMA (~2 h com 4 processos)
    python reproduzir.py --ogle               # + eventos reais do OGLE (precisa de internet)
    python reproduzir.py --etapas analises    # só uma etapa: dados, treinos, analises, benchmark, ogle
    python reproduzir.py --listar             # só mostra os comandos, sem rodar
    python reproduzir.py --rapido             # teste de fumaça: conjuntos pequenos e 2 épocas (NÃO usar no texto)

Cada passo é pulado se a saída já existir (use --forcar para refazer). Ao final grava models/MANIFESTO_final.json com o
commit do git, as sementes, os comandos e o hash SHA-256 de cada modelo e conjunto de dados.
Os conjuntos de dados só são aproveitados se tiverem sido gerados por ESTA versão do código e com a semente esperada (cada
arquivo guarda a versão do pré-processamento e a semente). Um arquivo antigo com o mesmo nome é regenerado, e os resultados
que dependiam dele são movidos para models/anterior_<data>/, para que nada de uma execução anterior se misture à nova.
Tempo aproximado em CPU: dados ~1 h, cada treino ~10 min (8 treinos), análises ~5 min, benchmark ~3 h (4 processos).
A tabela "o que gera o quê" está em docs/REPRODUZIR.md."""
import sys, os, json, time, shutil, hashlib, pathlib, argparse, subprocess, datetime, platform

ROOT = pathlib.Path(__file__).resolve().parent
PY = sys.executable
D = "data/processed"

ap = argparse.ArgumentParser()
ap.add_argument("--etapas", nargs="*", default=None, help="dados treinos analises benchmark ogle latex")
ap.add_argument("--benchmark", action="store_true"); ap.add_argument("--ogle", action="store_true")
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--n-benchmark", type=int, default=100)
ap.add_argument("--forcar", action="store_true"); ap.add_argument("--listar", action="store_true")
ap.add_argument("--rapido", action="store_true", help="teste de fumaça (conjuntos pequenos, 2 épocas); saídas com sufixo _rapido")
a = ap.parse_args()
etapas = a.etapas or (["dados", "treinos", "analises"] + (["benchmark"] if a.benchmark else []) + (["ogle"] if a.ogle else []) + ["latex"])

R = "_rapido" if a.rapido else ""
n = (lambda full, small: str(small if a.rapido else full))
EP = "6" if a.rapido else "30"
W = str(a.workers)

# ------------------------------------------------------------------------------------------------------------------
# Sementes: cada conjunto usa as sementes seed, seed+1, ..., seed+n−1 (um evento por semente), sem sobreposição.
# ------------------------------------------------------------------------------------------------------------------
DADOS = [
    # (saída, argumentos)
    (f"{D}/train_mix{R}.npz",       ["--n", n(30000, 600), "--caustic-frac", "0.7", "--seed", "42"]),
    (f"{D}/train_null{R}.npz",      ["--n", n(30000, 600), "--caustic-frac", "0.7", "--p-null", "0.2", "--seed", "100000"]),
    (f"{D}/test_uniform{R}.npz",    ["--n", n(6000, 300), "--caustic-frac", "0", "--seed", "900000"]),
    (f"{D}/test_null{R}.npz",       ["--n", n(3000, 200), "--p-null", "1", "--seed", "910000"]),
    (f"{D}/test_null_vbm{R}.npz",   ["--n", n(3000, 200), "--p-null", "1", "--null-vbm", "--seed", "920000"]),
    (f"{D}/test_null_indep{R}.npz", ["--n", n(3000, 200), "--p-null", "1", "--seed", "930000"]),
    # CONFIRMAÇÃO: priori uniforme, semente nova, avaliado UMA única vez, só pela rede final já escolhida.
    # Nenhuma decisão de projeto pode ser tomada olhando este conjunto.
    (f"{D}/test_confirmacao{R}.npz", ["--n", n(6000, 300), "--caustic-frac", "0", "--seed", "950000"]),
]
U = ["--uchan"]; X = ["--target", "x"]; DL = ["--delta"]
TU = ["--test-data", f"{D}/test_uniform{R}.npz", "--epochs", EP]
TREINOS = [
    # (nome do run, argumentos)                                       papel no texto
    (f"final{R}",        U + X + DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "0"]),      # rede principal
    (f"final_s1{R}",     U + X + DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "1"]),      # rede principal, 2ª semente
    (f"final_s2{R}",     U + X + DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "2"]),      # rede principal, 3ª semente
    (f"abl_sem_u{R}",    DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "0"]),             # ablação: sem u(τ), alvo |log s|
    (f"abl_sem_u_s1{R}", DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "1"]),
    (f"abl_sem_u_s2{R}", DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "2"]),
    (f"abl_u{R}",        U + DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "0"]),         # ablação: com u(τ), alvo |log s|
    (f"final_det{R}",    U + X + DL + ["--null", "--train-data", f"{D}/train_null{R}.npz", "--null-test", f"{D}/test_null{R}.npz", "--seed", "0"]),  # detecção
]
ANALISES = [
    # (saída que indica que já foi feito, script, argumentos)
    (f"models/rede_vs_pspl_final{R}.json",       "scripts/12_rede_vs_pspl.py", ["--run", f"final{R}", "--test", f"{D}/test_uniform{R}.npz"]),
    (f"models/rede_vs_pspl_abl_sem_u{R}.json",   "scripts/12_rede_vs_pspl.py", ["--run", f"abl_sem_u{R}", "--test", f"{D}/test_uniform{R}.npz"]),
    (f"models/r2_final{R}.json",                 "scripts/07_r2.py", ["--pred", f"{D}/test_pred_uniform_final{R}.npz", "--tag", f"final{R}"]),
    (f"models/r2_final_s1{R}.json",              "scripts/07_r2.py", ["--pred", f"{D}/test_pred_uniform_final_s1{R}.npz", "--tag", f"final_s1{R}"]),
    (f"models/r2_final_s2{R}.json",              "scripts/07_r2.py", ["--pred", f"{D}/test_pred_uniform_final_s2{R}.npz", "--tag", f"final_s2{R}"]),
    (f"models/rede_vs_pspl_final_s1{R}.json",    "scripts/12_rede_vs_pspl.py", ["--run", f"final_s1{R}", "--test", f"{D}/test_uniform{R}.npz"]),
    (f"models/rede_vs_pspl_final_s2{R}.json",    "scripts/12_rede_vs_pspl.py", ["--run", f"final_s2{R}", "--test", f"{D}/test_uniform{R}.npz"]),
    (f"models/rede_vs_pspl_final_det{R}.json",   "scripts/12_rede_vs_pspl.py", ["--run", f"final_det{R}", "--test", f"{D}/test_uniform{R}.npz"]),
    (f"models/r2_abl_sem_u{R}.json",             "scripts/07_r2.py", ["--pred", f"{D}/test_pred_uniform_abl_sem_u{R}.npz", "--tag", f"abl_sem_u{R}"]),
    (f"models/r2_abl_u{R}.json",                 "scripts/07_r2.py", ["--pred", f"{D}/test_pred_uniform_abl_u{R}.npz", "--tag", f"abl_u{R}"]),
    (f"models/analise_s_final{R}.json",          "scripts/06_analise_s.py", ["--run", f"final{R}"]),
    (f"models/regime_planetario_final{R}.json",  "scripts/15_regime_planetario.py", ["--run", f"final{R}"]),
    (f"models/deteccao_final_det{R}.json",       "scripts/10_deteccao.py", ["--run", f"final_det{R}", "--null-calib", f"{D}/test_null{R}.npz"]),
    (f"models/deteccao_final_det{R}_test_null_vbm{R}.json", "scripts/10_deteccao.py",
     ["--run", f"final_det{R}", "--null-file", f"{D}/test_null_vbm{R}.npz", "--model", f"models/cnn_final_det{R}.pt", "--null-calib", f"{D}/test_null{R}.npz"]),
    (f"models/deteccao_final_det{R}_logq-1.52.json", "scripts/10_deteccao.py",
     ["--run", f"final_det{R}", "--null-calib", f"{D}/test_null{R}.npz", "--logq-max", "-1.52"]),   # regime planetário (q < 0,03)
    (f"models/avaliacao_1l1s_final_det{R}.json", "scripts/11_avaliar_1l1s.py",
     ["--run", f"final_det{R}", "--null-calib", f"{D}/test_null{R}.npz", "--null-eval", f"{D}/test_null_indep{R}.npz"]),
    (f"models/confirmacao_final{R}.json",        "scripts/17_confirmacao.py", ["--run", f"final{R}", "--test", f"{D}/test_confirmacao{R}.npz"]),
    ("figures/fig_topologias.png",               "scripts/03b_figuras_cnn.py", []),
    ("figures/fig_degenerescencia_s.png",        "scripts/08_degenerescencia_s.py", []),
]
BENCH = [
    (f"{D}/benchmark_v2{R}/evt_{int(n(a.n_benchmark, 3)) - 1:05d}.json", "scripts/13_benchmark_pylima_v2.py",
     ["--n", n(a.n_benchmark, 3), "--workers", W, "--model", f"models/cnn_final{R}.pt", "--out", f"{D}/benchmark_v2{R}"]),
    (None, "scripts/14_resumo_benchmark_v2.py", ["--dir", f"{D}/benchmark_v2{R}"]),
]
OGLE = [(f"models/ogle_{ev}.json", "scripts/05_ogle.py", ["--event", ev, "--model", f"models/cnn_final{R}.pt"]) for ev in ("2012-BLG-0406", "2016-BLG-1195")]
LATEX = [(None, "scripts/16_latex.py", [])]

LOG = []
ARQUIVADO = [False]


def conjunto_ok(saida, args):
    """True se o arquivo existe e foi gerado por esta versão do pré-processamento, com a semente e o tamanho esperados."""
    f = ROOT / saida
    if not f.exists(): return False
    try:
        import numpy as np
        sys.path.insert(0, str(ROOT)); from mlens.features import VERSAO
        d = np.load(f)
        return ("versao" in d.files and int(d["versao"]) == VERSAO and int(d["semente"]) == int(args[args.index("--seed") + 1])
                and len(d["y"]) == int(args[args.index("--n") + 1]))
    except Exception:
        return False


def arquiva_resultados():
    """Move os resultados de execuções anteriores (modelos, previsões, métricas, benchmark) para uma pasta datada."""
    if ARQUIVADO[0] or a.listar: return
    ARQUIVADO[0] = True
    dest = ROOT / f"models/anterior_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    itens = [f for f in (ROOT / "models").glob("*") if f.is_file() and R in f.name and (f.suffix == ".json" or f.name.startswith(("cnn_final", "cnn_abl")))
             and (bool(R) or "_rapido" not in f.name)]
    itens += [f for f in (ROOT / D).glob("test_pred_*") if (R in f.name if R else "_rapido" not in f.name)]
    itens += [f for f in (ROOT / D).glob("test_null_pplanet_*") if (R in f.name if R else "_rapido" not in f.name)]
    b = ROOT / f"{D}/benchmark_v2{R}"
    if b.exists(): itens.append(b)
    if not itens: return
    dest.mkdir(parents=True, exist_ok=True)
    for f in itens: shutil.move(str(f), str(dest / f.name))
    print(f"[arquiva] {len(itens)} resultados de execuções anteriores movidos para {dest.relative_to(ROOT)}")


def roda(saida, script, args):
    cmd = [PY, script, *args]; txt = " ".join(["python", script, *args])
    if a.listar:
        print(txt); return
    if saida and (ROOT / saida).exists() and not (a.forcar and saida.startswith(D + "/") and saida.endswith(".npz")):
        print(f"[pula] {txt}   (já existe {saida})"); LOG.append(dict(comando=txt, status="pulado")); return
    print(f"\n[roda] {txt}", flush=True); t = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    LOG.append(dict(comando=txt, status="ok" if r.returncode == 0 else f"erro {r.returncode}", segundos=round(time.time() - t)))
    if r.returncode != 0:
        print(f"\nERRO em: {txt}\nCorrija e rode de novo o mesmo comando: os passos prontos serão pulados."); manifesto(); sys.exit(1)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def git(*args):
    try: return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception: return ""


def manifesto():
    if a.listar: return
    arq = [*(ROOT / "models").glob(f"cnn_*{R}.pt"), *[ROOT / s for s, _ in DADOS]]
    M = dict(data=datetime.datetime.now().isoformat(timespec="seconds"), git_commit=git("rev-parse", "HEAD"),
             git_tag=git("describe", "--tags", "--always"), alteracoes_nao_commitadas=bool(git("status", "--porcelain", "--", "mlens", "scripts")),
             python=platform.python_version(), plataforma=platform.platform(), rapido=a.rapido,
             sementes_dados={pathlib.Path(s).name: args[args.index("--seed") + 1] for s, args in DADOS},
             conjuntos_verificados={pathlib.Path(s).name: conjunto_ok(s, args) for s, args in DADOS},
             treinos={nome: " ".join(args) for nome, args in TREINOS}, comandos=LOG,
             sha256={str(p.relative_to(ROOT)): sha(p) for p in arq if p.exists()})
    (ROOT / "models").mkdir(exist_ok=True)
    out = ROOT / f"models/MANIFESTO_final{R}.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    M["comandos"] = old.get("comandos", []) + LOG
    out.write_text(json.dumps(M, indent=1, ensure_ascii=False), encoding="utf-8"); print(f"\nmanifesto: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    (ROOT / D).mkdir(parents=True, exist_ok=True); (ROOT / "models").mkdir(exist_ok=True); (ROOT / "figures").mkdir(exist_ok=True)
    if git("status", "--porcelain", "--", "mlens", "scripts") and not a.listar:
        print("AVISO: há alterações não commitadas em mlens/ ou scripts/. Para a versão da monografia, faça commit antes de rodar.")
    if a.forcar: arquiva_resultados()
    if "dados" in etapas:
        for saida, args in DADOS:
            if not a.listar and not conjunto_ok(saida, args):
                if (ROOT / saida).exists():
                    print(f"[antigo] {saida} não foi gerado por esta versão do código (ou tem outra semente): será regenerado.")
                    (ROOT / saida).unlink()
                arquiva_resultados()                       # o que dependia do conjunto antigo sai do caminho
            roda(saida, "scripts/01b_gerar_dataset_feat.py", [*args, "--workers", W, "--out", saida])
    elif not a.listar:
        ruins = [s_ for s_, args in DADOS if not conjunto_ok(s_, args)]
        if ruins and any(e in etapas for e in ("treinos", "analises")):
            sys.exit("ERRO: conjuntos ausentes ou de outra versão do código: " + ", ".join(ruins) + "\nRode primeiro:  python reproduzir.py --etapas dados")
    if "treinos" in etapas:
        for nome, args in TREINOS:
            roda(f"models/cnn_{nome}.pt", "scripts/02b_treinar_cnn.py", [*args, *TU, "--run", nome])
    if "analises" in etapas:
        for saida, script, args in ANALISES: roda(saida, script, args)
    if "benchmark" in etapas:
        for saida, script, args in BENCH: roda(saida, script, args)
    if "ogle" in etapas:
        for saida, script, args in OGLE: roda(saida, script, args)
    if "latex" in etapas and not a.rapido:
        for saida, script, args in LATEX: roda(saida, script, args)
    manifesto()
