# Question 3: two-asset efficient frontier in binomial world
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import save_table, save_fig

FIGS = Path(__file__).resolve().parents[1] / "figures"
DATA = Path(__file__).resolve().parents[1] / "data"
FIGS.mkdir(exist_ok=True)

u1, d1, u2, d2, p = 0.10, -0.12, 0.25, -0.20, 0.6

def mu(u,d,p): return p*u + (1-p)*d
def var(u,d,p): 
    m = mu(u,d,p)
    return p*(u-m)**2 + (1-p)*(d-m)**2

def cov12(u1,d1,u2,d2,p,rho):
    s1 = np.sqrt(var(u1,d1,p)); s2 = np.sqrt(var(u2,d2,p))
    return rho * s1 * s2

def frontier(rho):
    m1, m2 = mu(u1,d1,p), mu(u2,d2,p)
    v1, v2 = var(u1,d1,p), var(u2,d2,p)
    cov = cov12(u1,d1,u2,d2,p,rho)
    Sigma = np.array([[v1, cov],[cov, v2]])
    mu_vec = np.array([m1, m2])
    ones = np.ones(2)
    invS = np.linalg.inv(Sigma)
    A = ones @ invS @ ones
    B = ones @ invS @ mu_vec
    C = mu_vec @ invS @ mu_vec
    gmvp = (invS @ ones) / A
    # Sharpe/tangency with RF=0 proportional to invS * mu
    tang = (invS @ mu_vec) / B
    return gmvp, tang, (A,B,C,Sigma,mu_vec)

def scan_frontier(Sigma, mu_vec, w1_grid=np.linspace(-0.5,1.5,201)):
    ws = np.column_stack([w1_grid, 1-w1_grid])
    rets = ws @ mu_vec
    vars_ = np.einsum('ij,jk,ik->i', ws, Sigma, ws)
    return ws, rets, vars_

# Part 3: two rhos
rows=[]
for rho in (0.5, 0.0):
    gmvp, tang, (A,B,C,Sigma,mu_vec) = frontier(rho)
    rows.append({"rho":rho, "wGMV_1":gmvp[0], "wGMV_2":gmvp[1], "wSR_1":tang[0], "wSR_2":tang[1]})
pd.DataFrame(rows).to_csv(DATA/"q3_portfolios.csv", index=False)

# Part 5: plot efficient frontier at rho=0.5 and compare arbitrary w1 scan
gmvp, tang, (A,B,C,Sigma,mu_vec) = frontier(0.5)
ws, rets, vars_ = scan_frontier(Sigma, mu_vec)
fig, ax = plt.subplots()
ax.plot(vars_, rets)
ax.set_xlabel("Variance")
ax.set_ylabel("Mean Return")
ax.set_title("Mean-Variance Frontier (rho=0.5)")
save_fig(fig, FIGS/"q3_frontier_rho_0p5.png")

print(pd.DataFrame(rows))
