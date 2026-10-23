"""Monta o banner (90 x 120 cm, retrato) em PDF vetorial + PNG de pré-visualização.
Uso (na raiz do repo):  python banner/build_banner.py
Edite SÓ o bloco CONTEÚDO. Figuras vêm de figures/ e métricas de models/metrics.json."""
import json, sys, pathlib
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

BODY = ParagraphStyle("b", fontName="S", fontSize=30, leading=40, alignment=TA_JUSTIFY, textColor=black)
CAP = ParagraphStyle("c", parent=BODY, fontSize=22, leading=29, textColor=GREY)
REF = ParagraphStyle("r", parent=BODY, fontSize=23, leading=30, alignment=TA_LEFT)
BOX = ParagraphStyle("x", parent=BODY, fontSize=26, leading=35)
_m = json.load(open("models/metrics_cnn.json")); r2 = _m.get("dchi2>100", next(iter(_m.values())))
meta = json.load(open("models/meta_cnn.json"))   # n_train, n_test (escrito por 02b_treinar_cnn.py)
f2 = lambda x: f"{x:.2f}".replace(".", ",")

# ===================== CONTEÚDO (edite aqui) =====================
TITLE = ["Modelagem de Microlentes Gravitacionais com", "Redes Neurais: Aplicações em Sistemas 2L1S"]
AUTHORS = "<u><b>[Seu nome completo]</b></u>¹, [Coautor(es)]¹"
AFIL = "¹Universidade Federal do Ceará"
EVENT = "XLIX Reunião Anual da Sociedade Astronômica Brasileira"

INTRO = [
 "A microlente gravitacional ocorre quando uma estrela (lente) passa diante de uma estrela de fundo (fonte) e amplifica "
 "temporariamente seu brilho [1]. Se a lente possui um planeta, surgem anomalias breves na curva de luz, o que torna a "
 "técnica sensível a planetas de baixa massa e distantes de sua estrela [2,3].",
 "Em etapa anterior, modelamos eventos 1L1S (uma lente, uma fonte) com Redes Neurais Convolucionais (CNNs), recuperando "
 "t<sub>0</sub>, u<sub>0</sub> e t<sub>E</sub> com R² médio de 0,975, 0,945 e 0,987 (teste com 1169 curvas). "
 "Aqui, expandimos para lentes binárias (2L1S), em que a amplificação diverge sobre as <b>cáusticas</b>: curvas no plano da fonte "
 "cuja travessia produz picos intensos e que codificam q, s e α [3,4].",
 "Missões de grande volume de dados, como o Nancy Grace Roman Space Telescope, demandam métodos de inferência rápidos, "
 "complementares à inferência bayesiana tradicional.",
]
OBJ = ("Treinar uma CNN 1D para inferir os parâmetros físicos de eventos 2L1S — t<sub>0</sub>, u<sub>0</sub>, "
       "t<sub>E</sub>, razão de massa <i>q</i>, separação projetada <i>s</i> e ângulo de trajetória α — a partir de curvas de luz "
       "sintéticas realistas (lacunas, ruído gaussiano e múltiplos telescópios), como etapa para um benchmark com métodos bayesianos.")
FIG0_CAP = ("<b>Figura 1.</b> Topologias de cáustica (vermelho) para q = 10<super>-2</super> e a trajetória da fonte (azul): "
            "close, ressonante e wide, com as curvas de luz correspondentes. A travessia da cáustica gera os picos.")
FLOW = [
 ("#e8870e", "SIMULAÇÃO", "<b>1. Curvas 2L1S.</b> VBMicrolensing. Priori: t<sub>0</sub> ∈ [60, 140] d, u<sub>0</sub> ∈ [0,01; 1], t<sub>E</sub> ∈ [10, 50] d, "
  "log q ∈ [−4, −1], s ∈ [0,5; 2], α ∈ [0, 2π]; ρ = 10<super>-3</super>. 70% das trajetórias são ancoradas em pontos da cáustica "
  "(anomalia detectável); 30% seguem a priori uniforme."),
 ("#2e86c1", "EFEITOS OBSERVACIONAIS", "<b>2. Dados realistas.</b> Três sítios (Chile, África do Sul, Austrália), cadência de 1 h em janela noturna de 8 h; "
  "30% de noites perdidas; lacuna sazonal de até 30 d; ruído gaussiano σ = σ<sub>0</sub>√A (σ<sub>0</sub> = 1–2%)."),
 ("#3a9d4a", "PRÉ-PROCESSAMENTO", "<b>3. Entrada da rede.</b> Ajuste 1L1S (PSPL); o <b>resíduo</b> em z-score é binado em 256 pontos na escala de t<sub>E</sub>, "
  "com canais de fluxo, resíduo e máscara de lacunas, mais Δχ² como escalar."),
 ("#8e44ad", "MODELO", f"<b>4. CNN 1D.</b> 4 blocos Conv–BatchNorm–GELU–MaxPool + camada densa. Saídas: t<sub>0</sub>, u<sub>0</sub>, t<sub>E</sub>, log q, |log s| "
  f"(degenerescência s ↔ 1/s), (cos α, sin α) e sinal de log s. {meta['n_train']} curvas de treino (Δχ² &gt; 100), {meta['n_test']} de teste."),
 ("#c0392b", "AVALIAÇÃO", "<b>5. Métricas.</b> R² por parâmetro e erro angular médio de α, em eventos com anomalia detectável (Δχ² &gt; 100)."),
]
RES_TXT = (f"Em eventos com anomalia detectável (Δχ² &gt; 100; {r2['n']} curvas de teste), a CNN 1D obtém R² = {f2(r2['t0'])} em t<sub>0</sub>, "
           f"{f2(r2['u0'])} em u<sub>0</sub> e {f2(r2['tE'])} em t<sub>E</sub>. Para os parâmetros do sistema binário, R² = {f2(r2['logq'])} em log q e "
           f"{f2(r2['abs_logs'])} em |log s|; o ângulo α é recuperado com erro médio de {r2['alpha_erro_deg']:.0f}° (Fig. 3 e Tabela 1).")
FIG1_CAP = ("<b>Figura 2.</b> Evento 2L1S sintético (s = 1,3; q = 5×10<super>-3</super>) observado por três telescópios. O ajuste 1L1S absorve o pico principal; "
            "a anomalia da cáustica fica no <b>resíduo</b>, que é a entrada da rede.")
FIG2_CAP = f"<b>Figura 3.</b> Parâmetro previsto × verdadeiro no conjunto de teste com Δχ² &gt; 100. Linha tracejada: previsão perfeita."
CONC = ("<b>Estado atual.</b> Com entrada em resíduos, amostragem ancorada em cáusticas e CNN 1D, os parâmetros do evento principal são bem "
        "recuperados (R² &gt; 0,78) e os do sistema binário passam a ser parcialmente acessíveis: log q (R² = 0,77), |log s| (0,58) e α "
        "(erro médio de 36°, contra ~90° ao acaso). A diferença entre o erro de treino e o de teste indica que mais dados devem melhorar "
        "o desempenho. <b>Próximos passos:</b> (i) mais dados e regularização; (ii) saídas probabilísticas para as degenerescências "
        "(s ↔ 1/s, α); (iii) benchmark contra inferência bayesiana e pyLIMA, e eventos reais do OGLE.")
REFS = ["1. Paczyński, B. 1986, ApJ, 304, 1", "2. Mao, S. & Paczyński, B. 1991, ApJ, 374, L37",
        "3. Gould, A. & Loeb, A. 1992, ApJ, 396, 104", "4. Bennett, D. P. & Rhie, S. H. 1996, ApJ, 472, 660",
        "5. Bozza, V. 2010, MNRAS, 408, 2188 (VBBinaryLensing)", "6. Bachelet, E. et al. 2017, AJ, 154, 203 (pyLIMA)"]
# =================================================================

c = canvas.Canvas("banner/banner_SAB_v3.pdf", pagesize=(W, H))

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
y = img(XL, y - 4, "figures/fig_topologias.png", w=CW * 0.8); y = par(XL, y, FIG0_CAP, CAP, gap=26)
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
y = img(XR, y, "figures/fig_evento.png", w=CW); y = par(XR, y, FIG1_CAP, CAP, gap=22)
y = img(XR, y, "figures/fig_pred_vs_true_cnn.png", w=CW * 0.86); y = par(XR, y, FIG2_CAP, CAP, gap=22)
import os
rp = json.load(open("models/metrics_cnn_pylima.json")) if os.path.exists("models/metrics_cnn_pylima.json") else None   # opcional (02d)
def _row(lab, old, k):
    r = [lab, old, f2(r2[k])] + ([f2(rp[k])] if rp else []); return r
alf = lambda d: f"{d['alpha_erro_deg']:.0f}°"
rows = [["Parâmetro", "1L1S (CNN)", "2L1S (VBM)"] + (["2L1S (pyLIMA)"] if rp else []),
        _row("t0", "0,975", "t0"), _row("u0", "0,945", "u0"), _row("tE", "0,987", "tE"), _row("log q", "—", "logq"), _row("|log s|", "—", "abs_logs"),
        ["α (erro médio)", "—", alf(r2)] + ([alf(rp)] if rp else [])]
t = Table(rows, colWidths=([CW * .22, CW * .24, CW * .26, CW * .28] if rp else [CW * .3, CW * .33, CW * .37])); t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "S", 26), ("FONT", (0, 0), (-1, 0), "S-B", 26), ("BACKGROUND", (0, 0), (-1, 0), HexColor("#dbe7f7")),
    ("GRID", (0, 0), (-1, -1), 1, HexColor("#999999")), ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
tw, th = t.wrap(CW, 10 ** 6); t.drawOn(c, XR, y - th); y -= th + 12
y = par(XR, y, "<b>Tabela 1.</b> R² por parâmetro (1L1S: etapa anterior; 2L1S: este trabalho, Δχ² &gt; 100; pyLIMA: simulador independente, sem retreino).", CAP, gap=26)
y = title(XR, y, "Conclusão e perspectivas"); y = par(XR, y, CONC, gap=26)
y = title(XR, y, "Referências")
for r in REFS: y = par(XR, y, r, REF, gap=6)
yR = y

c.save(); print("fim coluna esquerda: %.0f mm acima do rodapé-limite; direita: %.0f mm" % ((yL - BOT) / mm, (yR - BOT) / mm))
