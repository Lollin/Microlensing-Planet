"""Reproduz TODOS os resultados da versão final da monografia, com sementes fixas.

    python reproduzir.py                      # dados + treinos + análises (sem benchmark e sem OGLE)
    python reproduzir.py --benchmark          # + benchmark com o pyLIMA (~2 h com 4 processos)
    python reproduzir.py --ogle               # + eventos reais do OGLE (precisa de internet)
    python reproduzir.py --etapas analises    # só uma etapa: dados, treinos, analises, benchmark, ogle
    python reproduzir.py --listar             # só mostra os comandos, sem rodar
    python reproduzir.py --rapido             # teste de fumaça: conjuntos pequenos e 2 épocas (NÃO usar no texto)

Cada passo é pulado se a saída já existir (use --forcar para refazer). Ao final grava models/MANIFESTO_final.json com o
commit do git, as sementes, os comandos e o hash SHA-256 de cada modelo e conjunto de dados.
Tempo aproximado em CPU (8 GB): dados ~1 h, cada treino ~25 min (6 treinos), análises ~5 min, benchmark ~2 h.
A tabela "o que gera o quê" está em docs/REPRODUZIR.md."""
import sys, os, json, time, hashlib, pathlib, argparse, subprocess, datetime, platform

ROOT = pathlib.Path(__file__).resolve().parent
PY = sys.executable
D = "data/processed"

ap = argparse.ArgumentParser()
ap.add_argument("--etapas", nargs="*", default=None, help="dados treinos analises benchmark ogle")
ap.add_argument("--benchmark", action="store_true"); ap.add_argument("--ogle", action="store_true")
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--n-benchmark", type=int, default=100)
ap.add_argument("--forcar", action="store_true"); ap.add_argument("--listar", action="store_true")
ap.add_argument("--rapido", action="store_true", help="teste de fumaça (conjuntos pequenos, 2 épocas); saídas com sufixo _rapido")
a = ap.parse_args()
etapas = a.etapas or (["dados", "treinos", "analises"] + (["benchmark"] if a.benchmark else []) + (["ogle"] if a.ogle else []))

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
]
U = ["--uchan"]; X = ["--target", "x"]; DL = ["--delta"]
TU = ["--test-data", f"{D}/test_uniform{R}.npz", "--epochs", EP]
TREINOS = [
    # (nome do run, argumentos)                                       papel no texto
    (f"final{R}",        U + X + DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "0"]),      # rede principal (Tabs. 3, 4)
    (f"final_s1{R}",     U + X + DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "1"]),      # 2ª semente (Tab. 12)
    (f"abl_sem_u{R}",    DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "0"]),             # ablação: sem u(τ), alvo |log s|
    (f"abl_u{R}",        U + DL + ["--train-data", f"{D}/train_mix{R}.npz", "--seed", "0"]),         # ablação: com u(τ), alvo |log s|
    (f"final_det{R}",    U + X + DL + ["--null", "--train-data", f"{D}/train_null{R}.npz", "--null-test", f"{D}/test_null{R}.npz", "--seed", "0"]),  # detecção
]
ANALISES = [
    # (saída que indica que já foi feito, script, argumentos)
    (f"models/rede_vs_pspl_final{R}.json",       "scripts/12_rede_vs_pspl.py", ["--run", f"final{R}"]),
    (f"models/rede_vs_pspl_abl_sem_u{R}.json",   "scripts/12_rede_vs_pspl.py", ["--run", f"abl_sem_u{R}"]),
    (f"models/r2_final{R}.json",                 "scripts/07_r2.py", ["--pred", f"{D}/test_pred_uniform_final{R}.npz", "--tag", f"final{R}"]),
    (f"models/r2_final_s1{R}.json",              "scripts/07_r2.py", ["--pred", f"{D}/test_pred_uniform_final_s1{R}.npz", "--tag", f"final_s1{R}"]),
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
    ("figures/fig_topologias.png",               "scripts/03b_figuras_cnn.py", []),
    ("figures/fig_degenerescencia_s.png",        "scripts/08_degenerescencia_s.py", []),
]
BENCH = [
    (f"{D}/benchmark_v2{R}/evt_{int(n(a.n_benchmark, 3)) - 1:05d}.json", "scripts/13_benchmark_pylima_v2.py",
     ["--n", n(a.n_benchmark, 3), "--workers", W, "--model", f"models/cnn_final{R}.pt", "--out", f"{D}/benchmark_v2{R}"]),
    (None, "scripts/14_resumo_benchmark_v2.py", ["--dir", f"{D}/benchmark_v2{R}"]),
]
OGLE = [(None, "scripts/05_ogle.py", ["--event", ev, "--model", f"models/cnn_final{R}.pt"]) for ev in ("2012-BLG-0406", "2016-BLG-1195")]

LOG = []


def roda(saida, script, args):
    cmd = [PY, script, *args]; txt = " ".join(["python", script, *args])
    if a.listar:
        print(txt); return
    if saida and (ROOT / saida).exists() and not a.forcar:
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
    if "dados" in etapas:
        for saida, args in DADOS:
            roda(saida, "scripts/01b_gerar_dataset_feat.py", [*args, "--workers", W, "--out", saida])
    if "treinos" in etapas:
        for nome, args in TREINOS:
            roda(f"models/cnn_{nome}.pt", "scripts/02b_treinar_cnn.py", [*args, *TU, "--run", nome])
    if "analises" in etapas:
        for saida, script, args in ANALISES: roda(saida, script, args)
    if "benchmark" in etapas:
        for saida, script, args in BENCH: roda(saida, script, args)
    if "ogle" in etapas:
        for saida, script, args in OGLE: roda(saida, script, args)
    manifesto()
