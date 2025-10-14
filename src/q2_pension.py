from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Salary base S0 can be normalized to 1 (it cancels in x). We keep it explicit for clarity.
S0   = 1.0
alpha = 0.50   # retirement payout as a fraction of salary
r     = 0.04   # annual interest (discrete) or force of interest (continuous) as context requires
g     = 0.01   # annual salary growth rate
n     = 40     # working years (contribution horizon)
tau   = 20     # retirement years (payout horizon)

# -----------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# ---------- Helpers: finite sums / integrals with edge-case handling ----------
def saving_factor_discrete(r: float, g: float, n: int) -> float:
    """
    A_n = sum_{t=1..n} (1+g)^{t-1} (1+r)^{n-t}
        = (1+r)^{n-1} * [ 1 - ((1+g)/(1+r))^n ] / [ 1 - (1+g)/(1+r) ]     if r != g
        = n * (1+r)^{n-1}                                                 if r == g
    This is the accumulation at time n of $1 of salary contributed each year end t=1..n.
    """
    if abs(r - g) < 1e-12:
        return n * (1 + r) ** (n - 1)
    ratio = (1 + g) / (1 + r)
    return (1 + r) ** (n - 1) * (1 - ratio ** n) / (1 - ratio)

def retirement_need_discrete(alpha: float, r: float, g: float, n: int, tau: int, S0: float = 1.0) -> float:
    """
    B_n = value at time n required to fund a retirement stream:
          payments at the end of years k=1..tau equal to alpha * S0 * (1+g)^{n+k-1}
    PV at n:
      B_n = alpha * S0 * (1+g)^{n-1} * sum_{k=1..tau} [ ((1+g)/(1+r))^k ]
          = alpha * S0 * (1+g)^{n-1} * ((1+g)/(1+r)) * [1 - ((1+g)/(1+r))^tau] / [1 - (1+g)/(1+r)]
    If r == g, limit gives:
      B_n = alpha * S0 * (1+g)^{n-1} * tau * (1+g)/(1+r) = alpha * S0 * (1+g)^{n-1} * tau
    """
    if abs(r - g) < 1e-12:
        return alpha * S0 * (1 + g) ** (n - 1) * tau
    ratio = (1 + g) / (1 + r)
    return alpha * S0 * (1 + g) ** (n - 1) * ratio * (1 - ratio ** tau) / (1 - ratio)

def x_discrete(alpha: float, r: float, g: float, n: int, tau: int, S0: float = 1.0) -> float:
    """
    Contribution rate x (as a fraction of salary) s.t.
      x * S0 * A_n  =  B_n   ⇒   x = B_n / (S0 * A_n)
    """
    A = saving_factor_discrete(r, g, n)
    B = retirement_need_discrete(alpha, r, g, n, tau, S0=S0)
    return (B / (S0 * A))

# ----- Continuous-time analog (continuous compounding & continuously paid retirement) -----
def saving_factor_continuous(r: float, g: float, n: float) -> float:
    """
    A_c = ∫_0^n S0 * e^{g t} * e^{r (n-t)} dt / S0
        = e^{r n} * ∫_0^n e^{(g-r) t} dt
        = e^{r n} * (e^{(g-r) n} - 1) / (g - r),   r != g
        = e^{r n} * n,                             r == g
    """
    if abs(r - g) < 1e-12:
        return np.exp(r * n) * n
    return np.exp(r * n) * (np.exp((g - r) * n) - 1) / (g - r)

def retirement_need_continuous(alpha: float, r: float, g: float, n: float, tau: float, S0: float = 1.0) -> float:
    """
    B_c = value at time n to fund a continuous retirement income:
          cashflow rate alpha * S0 * e^{g(n+u)} over u in [0, tau]
    PV at n:
      B_c = ∫_0^tau alpha * S0 * e^{g(n+u)} * e^{-r u} du
          = alpha * S0 * e^{g n} * ∫_0^tau e^{(g-r) u} du
          = alpha * S0 * e^{g n} * (e^{(g-r) tau} - 1)/(g - r),  r != g
          = alpha * S0 * e^{g n} * tau,                          r == g
    """
    if abs(r - g) < 1e-12:
        return alpha * S0 * np.exp(g * n) * tau
    return alpha * S0 * np.exp(g * n) * (np.exp((g - r) * tau) - 1) / (g - r)

def x_continuous(alpha: float, r: float, g: float, n: float, tau: float, S0: float = 1.0) -> float:
    A = saving_factor_continuous(r, g, n)
    B = retirement_need_continuous(alpha, r, g, n, tau, S0=S0)
    return (B / (S0 * A))

# ---------- Main computations ----------
xd = x_discrete(alpha, r, g, n, tau, S0=S0)
xc = x_continuous(alpha, r, g, n, tau, S0=S0)
xd_vs_xc = xd > xc

pd.DataFrame({
    "alpha":[alpha], "r":[r], "g":[g], "n":[n], "tau":[tau],
    "x_discrete":[xd], "x_continuous":[xc], "discrete_<_cont":[xd_vs_xc],
    "pct_diff": [100 * (xc - xd) / xd if xd != 0 else np.nan]
}).to_csv(DATA / "q2_x_values.csv", index=False)

# ----- Sensitivity over (g, r) grid for x_continuous (2.4) -----
g_vals = np.linspace(0.00, 0.10, 11)   # 0%..10%
r_vals = np.linspace(0.00, 0.10, 11)   # 0%..10%

Z = np.empty((len(g_vals), len(r_vals)))
for i, gg in enumerate(g_vals):
    for j, rr in enumerate(r_vals):
        Z[i, j] = x_continuous(alpha, rr, gg, n, tau, S0=S0)

# Save the grid as a tidy CSV
rows = []
for i, gg in enumerate(g_vals):
    for j, rr in enumerate(r_vals):
        rows.append({"g": float(gg), "r": float(rr), "x": float(Z[i, j])})
pd.DataFrame(rows).to_csv(DATA / "q2_sensitivity.csv", index=False)

# Heatmap (no seaborn)
fig, ax = plt.subplots()
im = ax.imshow(Z, origin="lower", aspect="auto",
               extent=[r_vals.min(), r_vals.max(), g_vals.min(), g_vals.max()])
ax.set_xlabel("r")
ax.set_ylabel("g")
ax.set_title("Contribution rate x (continuous) across g and r")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("x (fraction of salary)")
fig.tight_layout()
fig.savefig(FIGS / "q2_sensitivity_heatmap.png", dpi=160, bbox_inches="tight")

print({
    "x_discrete": xd,
    "x_continuous": xc,
    "pct_diff": float(100 * (xc - xd) / xd) if xd != 0 else np.nan
})
