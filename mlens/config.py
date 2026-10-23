"""Configuração central. Mude AQUI e tudo (dataset, treino, figuras) acompanha."""
import numpy as np

# Janela de observação (dias, tempo relativo)
T_MIN, T_MAX = 0.0, 200.0
N_BINS = 256                      # resolução da grade de entrada da rede

# Prioris dos parâmetros físicos (amostragem uniforme, exceto onde indicado)
PRIORS = {
    "t0":   (60.0, 140.0),        # dias
    "u0":   (0.01, 1.0),          # raios de Einstein
    "tE":   (10.0, 50.0),         # dias
    "logq": (-4.0, -1.0),         # log10 da razão de massa (planeta -> ~1e-4..1e-3)
    "logs": (np.log10(0.5), np.log10(2.0)),  # log10 da separação projetada
    "alpha": (0.0, 2 * np.pi),    # rad
}
RHO = 1e-3                        # raio da fonte (fixo nesta etapa)

# Rede de telescópios (offset de fase diário em horas, cadência em horas, ruído relativo)
SITES = {
    "Chile":        {"phase_h": 0.0,  "cadence_h": 1.0, "sigma": 0.010},
    "Africa do Sul": {"phase_h": 8.0,  "cadence_h": 1.0, "sigma": 0.015},
    "Australia":    {"phase_h": 16.0, "cadence_h": 1.0, "sigma": 0.020},
}
NIGHT_HOURS = 8.0                 # janela diária de observação por sítio
P_BAD_WEATHER = 0.30              # prob. de perder uma noite inteira
SEASON_GAP_DAYS = (0.0, 30.0)     # lacuna contígua aleatória (comprimento)

# Alvos que a rede prevê. alpha entra como (cos, sin) para evitar descontinuidade 0/2pi.
TARGETS = ["t0", "u0", "tE", "logq", "logs", "alpha"]
SEED = 42

# Fração de eventos cuja trajetória é FORÇADA a passar pela cáustica (resto: priori uniforme).
# Motivo: com priori uniforme ~metade dos eventos não tem anomalia detectável -> q, s, α não têm sinal.
CAUSTIC_FRAC = 0.7
U0_MAX_CAUSTIC = 1.5
