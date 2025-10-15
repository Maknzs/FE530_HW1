from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# ---------------- Inputs ----------------
S0 = 100.0              # today's spot
RF = 0.04               # monthly risk-free rate (as given in the prompt)
F_candidates = [104.0, 105.0, 103.0]

# Try to reuse Q1 binomial params for u,d,pi (not strictly required for pricing)
try:
    q1 = pd.read_csv(DATA / "q1_binomial_params.csv")
    pi = float(q1.loc[0, "pi"])
    u  = float(q1.loc[0, "u"])
    d  = float(q1.loc[0, "d"])
except Exception:
    # sensible fallback if file missing
    pi, u, d = 0.6, 0.03, -0.02

U, D = 1.0 + u, 1.0 + d
S_up, S_down = S0 * U, S0 * D

# ---------------- Fair forward price ----------------
F_fair = S0 * (1.0 + RF)
pd.DataFrame({"S0":[S0], "RF":[RF], "F_fair":[F_fair]}).to_csv(DATA/"q4_fair.csv", index=False)

# ---------------- Risk-neutral probability and check ----------------
# q = ((1+RF) - D) / (U - D), E^Q[S1] = S0*(qU + (1-q)D) = S0*(1+RF)
q = ((1.0 + RF) - D) / (U - D)
valid_q = (0.0 <= q <= 1.0)
E_Q_S1 = S0 * (q * U + (1.0 - q) * D)

pd.DataFrame({
    "S0":[S0], "U":[U], "D":[D], "RF":[RF], "q":[q],
    "valid_q":[valid_q], "E_Q_S1":[E_Q_S1], "S0*(1+RF)":[S0 * (1.0 + RF)]
}).round(6).to_csv(DATA/"q4_rn_check.csv", index=False)

# ---------------- Long-forward payoff per state ----------------
def payoff_table(F):
    return pd.DataFrame({
        "state": ["up","down"],
        "prob":  [pi, 1.0-pi],
        "S1":    [S_up, S_down],
        "F0":    [F, F],
        "payoff_long_forward": [S_up - F, S_down - F]
    })

payoff_table(F_fair).round(6).to_csv(DATA/"q4_payoff_fair.csv", index=False)

# ---------------- Mispricing cases (cash-and-carry vs reverse) ----------------
rows = []
for F in F_candidates:
    mis = F - F_fair
    if mis > 1e-12:
        direction = "short forward (overpriced); cash-and-carry"
    elif mis < -1e-12:
        direction = "long forward (underpriced); reverse cash-and-carry"
    else:
        direction = "fair (no arbitrage)"
    rows.append({"F": F, "F_fair": F_fair, "mispricing": mis, "strategy": direction})

pd.DataFrame(rows).round(6).to_csv(DATA/"q4_cases.csv", index=False)

print("Saved:", DATA/"q4_fair.csv")
print("Saved:", DATA/"q4_rn_check.csv")
print("Saved:", DATA/"q4_payoff_fair.csv")
print("Saved:", DATA/"q4_cases.csv")
