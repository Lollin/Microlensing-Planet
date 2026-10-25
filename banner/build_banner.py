"""Monta o banner (90 x 120 cm, retrato) em PDF vetorial + PNG de pré-visualização.
Uso (na raiz do repo):  python banner/build_banner.py
Edite SÓ o bloco CONTEÚDO. Figuras vêm de figures/ e métricas de models/metrics.json."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader

import matplotlib; D = matplotlib.get_data_path() + "/fonts/ttf/"
for n, f in [("S", "DejaVuSans"), ("S-B", "DejaVuSans-Bold"), ("S-I", "DejaVuSans-Oblique"), ("S-BI", "DejaVuSans-BoldOblique"),
             ("R", "DejaVuSerif"), ("R-B", "DejaVuSerif-Bold"), ("R-I", "DejaVuSerif-Italic"), ("R-BI", "DejaVuSerif-BoldItalic")]:
    pdfmetrics.registerFont(TTFont(n, D + f + ".ttf"))
pdfmetrics.registerFontFamily("S", normal="S", bold="S-B", italic="S-I", boldItalic="S-BI")
pdfmetrics.registerFontFamily("R", normal="R", bold="R-B", italic="R-I", boldItalic="R-BI")

W, H = 900 * mm, 1200 * mm
BLUE = HexColor("#1b4fa8"); GREY = HexColor("#444444")
M = 40 * mm; GAP = 30 * mm; CW = (W - 2 * M - GAP) / 2
XL, XR = M, M + CW + GAP
HEAD, FOOT = 125 * mm, 55 * mm
TOP = H - HEAD - 20 * mm; BOT = FOOT + 15 * mm

BODY = ParagraphStyle("b", fontName="S", fontSize=27, leading=36, alignment=TA_JUSTIFY, textColor=black)
CAP = ParagraphStyle("c", parent=BODY, fontSize=20, leading=26, textColor=GREY)
REF = ParagraphStyle("r", parent=BODY, fontSize=21, leading=27, alignment=TA_LEFT)
BOX = ParagraphStyle("x", parent=BODY, fontSize=24, leading=32)
_m = json.load(open("models/metrics_cnn.json")); meta = json.load(open("models/meta_cnn.json"))
POP = "uniform" if "uniform" in _m else "mix"            # população de teste: priori uniforme (preferida) ou 'mix'
R = _m[POP]; ALL, EXC = R["all"], R["exc>100"]
nT = ALL["n"]; frac_exc = 100 * EXC["n"] / nT
fm = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")
cell = lambda s, k, d=2: f"{fm(s[k]['med_abs'], d)} ({fm(s[k]['sigma68'], d)})"
POP_TXT = ("sorteados da priori uniforme, sem trajetórias ancoradas em cáusticas" if POP == "uniform"
           else "da mesma distribuição do treino, com trajetórias ancoradas em cáusticas (população enviesada)")

_y = np.load(f"data/processed/test_pred_{POP}.npz")["y"]
_md = lambda v: float(np.median(np.abs(v - np.median(v))))
BASE = {"t0": _md(_y[:, 0]), "u0": _md(_y[:, 1]), "tE_pct": float(np.median(100 * np.abs(_y[:, 2] - np.median(_y[:, 2])) / _y[:, 2])),
        "logq": _md(_y[:, 3]), "logs": _md(_y[:, 4]), "alpha_deg": 90.0}      # erro de prever sempre a mediana da priori (em α, 90° vale para ângulo uniforme)
rel = lambda k: 100 * ALL[k]["med_abs"] / BASE[k]

# ===================== CONTEÚDO (edite aqui) =====================
TITLE = ["Modelagem de Microlentes Gravitacionais com", "Redes Neurais: Aplicações em Sistemas 2L1S"]
AUTHORS = "<u><b>[Seu nome completo]</b></u>¹, [Coautor(es)]¹"
AFIL = "¹Universidade Federal do Ceará"
EVENT = "XLIX Reunião Anual da Sociedade Astronômica Brasileira"

INTRO = [
 "Em um evento de microlente gravitacional, uma estrela em primeiro plano amplifica por algum tempo o brilho de uma estrela ao fundo [1]. "
 "Quando a lente tem um planeta, surge uma anomalia curta na curva de luz. Por isso o método é sensível a planetas de baixa massa e a órbitas distantes [2,3].",
 "Em uma etapa anterior, redes neurais convolucionais foram aplicadas a eventos com uma lente e uma fonte (1L1S) e estimaram t<sub>0</sub>, u<sub>0</sub> e t<sub>E</sub> "
 "com R² de 0,975, 0,945 e 0,987 em 1169 curvas de teste. Agora tratamos sistemas com duas lentes (2L1S). Nesse caso a curva depende também da razão de massa q, "
 "da separação projetada s e do ângulo α da trajetória da fonte [3,4].",
 "Em lentes binárias, a amplificação diverge sobre as cáusticas, curvas no plano da fonte que produzem os picos mais intensos (Fig. 1). "
 "A forma das cáusticas muda com s e q, e é isso que permite estimar esses parâmetros a partir da curva de luz [4]. "
 "Missões com grande volume de dados, como o Roman Space Telescope, tornam útil um método de inferência rápido.",
]
OBJ = ("Verificar até que ponto uma rede neural convolucional unidimensional estima os parâmetros t<sub>0</sub>, u<sub>0</sub>, t<sub>E</sub>, q, s e α de eventos 2L1S simulados, "
       "com lacunas nos dados, ruído gaussiano e vários telescópios, e medir o erro em cada parâmetro. Esta etapa prepara a comparação com a inferência bayesiana.")
FIG0_CAP = ("<b>Figura 1.</b> Topologias de cáustica (vermelho) para q = 0,01 e trajetória da fonte (azul) nos regimes close (s &lt; 1), ressonante (s próximo de 1) e wide (s &gt; 1), "
            "com as curvas de luz correspondentes.")
FLOW = [
 ("#e8870e", "SIMULAÇÃO", "<b>1. Curvas 2L1S.</b> Geradas com o VBMicrolensing. Os parâmetros são sorteados de prioris uniformes: t<sub>0</sub> entre 60 e 140 d, u<sub>0</sub> entre 0,01 e 1, "
  "t<sub>E</sub> entre 10 e 50 d, log q entre -4 e -1, s entre 0,5 e 2 e α entre 0 e 2π. O raio da fonte é fixo em 0,001. No treino, 70% das trajetórias são forçadas a cruzar "
  "uma cáustica, para haver mais eventos com anomalia. O teste usa a priori uniforme."),
 ("#2e86c1", "OBSERVAÇÕES", "<b>2. Efeitos observacionais.</b> Três sítios simulados (Chile, África do Sul e Austrália), com uma medida por hora em janelas noturnas de 8 h, "
  "30% de noites perdidas, uma lacuna sazonal de até 30 d e ruído gaussiano de 1% a 2% do fluxo."),
 ("#3a9d4a", "ENTRADA", "<b>3. Entrada da rede.</b> Ajustamos um modelo 1L1S à curva. O resíduo normalizado do ajuste, em função de (t - t<sub>0</sub>)/t<sub>E</sub>, é agrupado em 256 intervalos "
  "e entra na rede com o fluxo e uma máscara das lacunas. Entram também quatro números do ajuste 1L1S (t<sub>E</sub>, u<sub>0</sub>, fluxo da fonte e t<sub>0</sub>). "
  "Nenhum parâmetro 2L1S verdadeiro é usado na entrada."),
 ("#8e44ad", "MODELO", f"<b>4. CNN 1D.</b> Quatro blocos convolucionais e uma camada densa. A rede estima t<sub>0</sub>, u<sub>0</sub>, t<sub>E</sub>, log q, o módulo de log s e α (por cos α e sen α). "
  f"Uma saída separada dá a probabilidade de s &gt; 1, e dela vem o sinal de log s. Assim a rede não assume que s e 1/s são equivalentes. "
  f"Treino com {meta['n_train']} curvas e validação com {meta['n_val']}; usamos a época de menor perda na validação."),
 ("#c0392b", "AVALIAÇÃO", f"<b>5. Métricas de erro.</b> Erro absoluto de cada parâmetro em {nT} curvas de teste, separadas pelo excesso de χ² do ajuste 1L1S "
  "(quanto a anomalia aparece nos dados). Em α usamos o erro angular. O sinal de log s é avaliado como classificação."),
]
RES_TXT = (f"Avaliamos a rede em {nT} curvas simuladas, com parâmetros {POP_TXT}. Em {fm(frac_exc, 0)}% delas o excesso de χ² do ajuste 1L1S passa de 100, ou seja, a anomalia aparece nos dados. "
           f"No conjunto todo, o erro mediano é de {fm(ALL['logq']['med_abs'])} dex em log q (prever a mediana da priori daria {fm(BASE['logq'])} dex), "
           f"{fm(ALL['logs']['med_abs'])} dex em log s (referência: {fm(BASE['logs'])} dex) e {fm(ALL['alpha_deg']['med_abs'], 0)}° em α (referência: 90°). "
           f"O sinal de log s é acertado em {fm(100 * ALL['sign']['acc'], 0)}% dos eventos (AUC = {fm(ALL['sign']['auc'])}; ao acaso, 50% e 0,50). "
           f"Entre os eventos com excesso de χ² acima de 100, os erros em log q, log s e α são menores (Tabela 1, Fig. 3).")
FIG1_CAP = ("<b>Figura 2.</b> Evento 2L1S simulado (s = 1,3; q = 0,005) observado por três telescópios simulados. A linha laranja é o melhor ajuste 1L1S, que descreve o pico principal. "
            "O resíduo desse ajuste, onde aparece a anomalia, é a entrada da rede.")
FIG2_CAP = (f"<b>Figura 3.</b> Resultados em {nT} curvas de teste. Em cima: log q e log s previstos contra os verdadeiros (cor indica a densidade de eventos; linha tracejada, previsão perfeita) "
            "e distribuição do erro angular de α. Embaixo: matriz de confusão do sinal de log s (porcentagem de cada classe verdadeira) e erro absoluto mediano em log q e em α, "
            "por faixa de excesso de χ² (barras entre os percentis 16 e 84).")
TAB_CAP = ("<b>Tabela 1.</b> Erro absoluto mediano e, entre parênteses, σ<sub>68</sub> (metade da largura do intervalo entre os percentis 16 e 84), em curvas simuladas de teste"
           + (" com priori uniforme" if POP == "uniform" else "") + ". A última coluna é o erro mediano de prever sempre a mediana da priori. Para o sinal de log s, taxa de acerto e AUC.")
CONC = (f"Em dados simulados {'com priori uniforme' if POP == 'uniform' else ''}, o erro mediano é de {fm(ALL['t0']['med_abs'])} d em t<sub>0</sub>, {fm(ALL['u0']['med_abs'], 3)} em u<sub>0</sub> e "
        f"{fm(ALL['tE_pct']['med_abs'], 1)}% em t<sub>E</sub>. Nos parâmetros do sistema binário, é de {fm(ALL['logq']['med_abs'])} dex em log q, {fm(ALL['logs']['med_abs'])} dex em log s e "
        f"{fm(ALL['alpha_deg']['med_abs'], 0)}° em α. Em relação à referência (prever a mediana da priori), esses erros equivalem a {fm(rel('logq'), 0)}% em log q, {fm(rel('logs'), 0)}% em log s e "
        f"{fm(rel('alpha_deg'), 0)}% em α. O sinal de log s é acertado em {fm(100 * ALL['sign']['acc'], 0)}% dos casos, contra 50% ao acaso. "
        f"A perda de validação deixou de cair após {meta['best_epoch'] + 1} das {meta['epochs']} épocas, enquanto a de treino continuou caindo; ainda não sabemos se mais dados ou mais regularização resolvem isso. "
        "Os resultados valem para as condições simuladas descritas e a rede não foi testada com dados reais.<br/><br/>"
        "<b>Perspectivas.</b> Medir o efeito do tamanho do conjunto de treino, tratar as degenerescências com saídas probabilísticas, "
        "comparar com a inferência bayesiana e com curvas geradas pelo pyLIMA, e aplicar a rede a eventos reais do OGLE.")
REFS = ["1. Paczyński, B. 1986, ApJ, 304, 1", "2. Mao, S. & Paczyński, B. 1991, ApJ, 374, L37",
        "3. Gould, A. & Loeb, A. 1992, ApJ, 396, 104", "4. Bennett, D. P. & Rhie, S. H. 1996, ApJ, 472, 660",
        "5. Bozza, V. 2010, MNRAS, 408, 2188"]
# =================================================================

c = canvas.Canvas("banner/banner_SAB.pdf", pagesize=(W, H))

def title(x, y, t):
    c.setFont("R-B", 60); c.setFillColor(black); c.drawString(x, y - 46, t)
    c.setStrokeColor(BLUE); c.setLineWidth(3); c.line(x, y - 62, x + CW, y - 62); return y - 86

def par(x, y, text, st=BODY, w=CW, gap=18):
    p = Paragraph(text, st); _, h = p.wrap(w, 10 ** 6); p.drawOn(c, x, y - h); return y - h - gap

def img(x, y, path, w=CW, gap=10):
    iw, ih = ImageReader(path).getSize(); h = w * ih / iw; c.drawImage(path, x + (CW - w) / 2, y - h, w, h); return y - h - gap

# cabeçalho / rodapé
c.setFillColor(BLUE); c.rect(0, H - HEAD, W, HEAD, stroke=0, fill=1); c.rect(0, 0, W, FOOT, stroke=0, fill=1)
c.setFillColor(white); c.setFont("R-B", 66)
c.drawCentredString(W / 2, H - 45 * mm, TITLE[0]); c.drawCentredString(W / 2, H - 70 * mm, TITLE[1])
for txt, yy, f in [(AUTHORS, H - 80 * mm, 40), (AFIL, H - 96 * mm, 36)]:
    p = Paragraph(txt, ParagraphStyle("a", fontName="R", fontSize=f, leading=f * 1.2, textColor=white, alignment=1))
    pw, ph = p.wrap(W - 240 * mm, 200); p.drawOn(c, 120 * mm, yy - ph)
for x0, lab in [(M, "LOGO\nUFC"), (W - M - 48 * mm, "LOGO\nGRUPO")]:       # placeholders de logos
    c.setStrokeColor(white); c.setDash(6, 6); c.rect(x0, H - HEAD + 22 * mm, 48 * mm, 72 * mm, stroke=1, fill=0); c.setDash()
    c.setFont("S", 22); [c.drawCentredString(x0 + 24 * mm, H - HEAD + 62 * mm - 28 * i, s) for i, s in enumerate(lab.split("\n"))]
c.setFont("R-B", 56); c.drawCentredString(W / 2, FOOT / 2 - 10, EVENT)
c.setFont("S", 22); c.drawString(M, FOOT / 2 - 8, "[logo SAB]"); c.drawRightString(W - M, FOOT / 2 - 8, "[logo CNPq]")

# ---------- coluna esquerda ----------
y = TOP; y = title(XL, y, "Introdução")
for t in INTRO: y = par(XL, y, t)
y = title(XL, y - 6, "Objetivo"); y = par(XL, y, OBJ)
y = img(XL, y - 4, "figures/fig_topologias.png", w=CW * 0.72); y = par(XL, y, FIG0_CAP, CAP, gap=26)
y = title(XL, y, "Metodologia")
for i, (col, tag, txt) in enumerate(FLOW):
    p = Paragraph(txt, BOX); _, h = p.wrap(CW - 70, 10 ** 6); bh = h + 74
    c.setFillColor(HexColor(col)); c.rect(XL, y - 40, 26, 26, stroke=0, fill=1)
    c.setFillColor(black); c.setFont("S-B", 27); c.drawString(XL + 40, y - 36, tag)
    c.setStrokeColor(HexColor(col)); c.setLineWidth(3); c.roundRect(XL, y - 54 - h - 14, CW, h + 14 + 14, 12, stroke=1, fill=0)
    p.drawOn(c, XL + 35, y - 54 - h)
    y -= (54 + h + 28)
    if i < len(FLOW) - 1:
        c.setStrokeColor(GREY); c.setLineWidth(4); c.line(XL + CW / 2, y + 2, XL + CW / 2, y - 26)
        c.setFillColor(GREY); p_ = c.beginPath(); p_.moveTo(XL + CW / 2 - 12, y - 18); p_.lineTo(XL + CW / 2 + 12, y - 18); p_.lineTo(XL + CW / 2, y - 34); p_.close(); c.drawPath(p_, fill=1, stroke=0)
        y -= 40
yL = y

# ---------- coluna direita ----------
y = TOP; y = title(XR, y, "Resultados"); y = par(XR, y, RES_TXT)
y = img(XR, y, "figures/fig_evento.png", w=CW * 0.64); y = par(XR, y, FIG1_CAP, CAP, gap=22)
y = img(XR, y, "figures/fig_resultados_cnn.png", w=CW * 0.68); y = par(XR, y, FIG2_CAP, CAP, gap=22)
hdr = [f"Todos (n = {nT})", f"Excesso > 100 (n = {EXC['n']})", "Referência"]
rows = [["Parâmetro", *hdr],
        ["t0 (dias)", cell(ALL, "t0"), cell(EXC, "t0"), fm(BASE["t0"])], ["u0", cell(ALL, "u0", 3), cell(EXC, "u0", 3), fm(BASE["u0"], 3)],
        ["tE (%)", cell(ALL, "tE_pct", 1), cell(EXC, "tE_pct", 1), fm(BASE["tE_pct"], 1)], ["log q (dex)", cell(ALL, "logq"), cell(EXC, "logq"), fm(BASE["logq"])],
        ["log s (dex)", cell(ALL, "logs"), cell(EXC, "logs"), fm(BASE["logs"])], ["α (graus)", cell(ALL, "alpha_deg", 0), cell(EXC, "alpha_deg", 0), "90"],
        ["Sinal de log s: acerto", f"{fm(100 * ALL['sign']['acc'], 0)}%", f"{fm(100 * EXC['sign']['acc'], 0)}%", "50%"],
        ["Sinal de log s: AUC", fm(ALL["sign"]["auc"]), fm(EXC["sign"]["auc"]), "0,50"]]
t = Table(rows, colWidths=[CW * .30, CW * .25, CW * .28, CW * .17]); t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "S", 21), ("FONT", (0, 0), (-1, 0), "S-B", 19), ("BACKGROUND", (0, 0), (-1, 0), HexColor("#dbe7f7")),
    ("GRID", (0, 0), (-1, -1), 1, HexColor("#999999")), ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
tw, th = t.wrap(CW, 10 ** 6); t.drawOn(c, XR, y - th); y -= th + 12
y = par(XR, y, TAB_CAP, CAP, gap=26)
y = title(XR, y, "Conclusão e perspectivas"); y = par(XR, y, CONC, gap=26)
y = title(XR, y, "Referências")
for r in REFS: y = par(XR, y, r, REF, gap=6)
yR = y

import re
clean = lambda s: re.sub(r"<[^>]+>", "", s.replace("<br/>", "\n")).replace("&gt;", ">").replace("&lt;", "<")
with open("banner/textos_banner.txt", "w", encoding="utf-8") as f:
    f.write("TÍTULO\n" + "\n".join(TITLE) + "\n\n")
    for nome, txt in [("INTRODUÇÃO", "\n\n".join(INTRO)), ("OBJETIVO", OBJ), ("RESULTADOS", RES_TXT), ("CONCLUSÃO E PERSPECTIVAS", CONC)]:
        f.write(nome + "\n" + clean(txt) + "\n\n")
    f.write("METODOLOGIA\n" + "\n\n".join(clean(t_) for _, _, t_ in FLOW) + "\n\n")
    f.write("LEGENDAS\n" + "\n\n".join(clean(x) for x in [FIG0_CAP, FIG1_CAP, FIG2_CAP, TAB_CAP]) + "\n\n")
    f.write("TABELA 1 (separada por tab)\n" + "\n".join("\t".join(r) for r in rows) + "\n\nREFERÊNCIAS\n" + "\n".join(REFS) + "\n")

c.save(); print("fim coluna esquerda: %.0f mm acima do rodapé-limite; direita: %.0f mm" % ((yL - BOT) / mm, (yR - BOT) / mm))
