# Question 2 pension math + sensitivity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import save_table, save_fig

FIGS = Path(__file__).resolve().parents[1] / "figures"
DATA = Path(__file__).resolve().parents[1] / "data"
FIGS.mkdir(exist_ok=True)

def x_discrete(alpha, r, g, n, tau):
    # end-of-year contributions, salary S_t = S0 (1+g)^t
    # derive from growing annuities:
    # Accumulated savings at n: x*S0 * sum_{t=1..n} (1+g)^t (1+r)^{n-t}
    A = (1+r)**n * ( (1+g)/(1+r) ) * ( 1 - ((1+g)/(1+r))**n ) / ( 1 - ( (1+g)/(1+r) ) )
    # PV at n of retirement stream: alpha*S0*(1+g)^n * sum_{k=1..tau} (1+g)^{k-1} / (1+r)^{k}
    B = (1+g)**n * ( 1 - ((1+g)/(1+r))**tau ) / ( (1+r) * ( 1 - ( (1+g)/(1+r) ) ) )
    return alpha * B / A

def x_continuous(alpha, r, g, n, tau):
    # limit m->infty of discrete with theta = (1+g/m)/(1+r/m)
    # gives integrals with continuous compounding:
    # savings at n: x*S0 * ∫_0^n e^{g t} e^{r (n-t)} dt = x*S0 e^{rn} * (e^{(g-r)n}-1)/(g-r)
    if abs(g - r) < 1e-12:
        A = np.exp(r*n) * n
    else:
        A = np.exp(r*n) * (np.exp((g-r)*n)-1)/(g-r)
    # PV at n of pension: alpha*S0 e^{g n} * ∫_0^tau e^{(g-r)u} du / e^{r}?
    if abs(g - r) < 1e-12:
        B = np.exp(g*n) * tau / np.exp(0) / (1+r)  # rough alignment with discrete, but use exact integral below
    # exact continuous version of growing annuity paid continuously over [0,tau]:
    if abs(g - r) < 1e-12:
        C = np.exp(g*n) * tau
    else:
        C = np.exp(g*n) * (np.exp((g-r)*tau)-1)/(g-r)
    # But retirement cashflows are received while funds remain invested; funding needed at n is alpha*S0*C
    return alpha * C / A

# Parameters from table
alpha=0.50; r=0.04; g=0.01; n=40.0; tau=20.0
xd = x_discrete(alpha,r,g,n,tau)
xc = x_continuous(alpha,r,g,n,tau)

pd.DataFrame({"x_discrete":[xd], "x_continuous":[xc]}).to_csv(DATA/"q2_x_values.csv", index=False)

# Sensitivity grid
grid = []
vals = [i/100 for i in range(1,11)]
for gg in vals:
    for rr in vals:
        grid.append((gg, rr, x_continuous(alpha, rr, gg, n, tau)))
Z = pd.DataFrame(grid, columns=["g","r","x"])
pivot = Z.pivot(index="g", columns="r", values="x")
save_table(pivot, DATA/"q2_sensitivity.csv")

# heatmap without seaborn
fig, ax = plt.subplots()
im = ax.imshow(pivot.values, origin="lower", aspect="auto")
ax.set_xticks(range(len(vals))); ax.set_xticklabels([f"{v:.2f}" for v in vals], rotation=45)
ax.set_yticks(range(len(vals))); ax.set_yticklabels([f"{v:.2f}" for v in vals])
ax.set_xlabel("r"); ax.set_ylabel("g"); ax.set_title("x sensitivity (continuous)")
fig.colorbar(im, ax=ax)
save_fig(fig, FIGS/"q2_sensitivity_heatmap.png")

print({"x_discrete": xd, "x_continuous": xc})
