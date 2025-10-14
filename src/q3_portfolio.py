from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# ---------- helpers ----------
def safe_inv(mat: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat)

def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights (sum to 1)."""
    ones = np.ones((Sigma.shape[0], 1))
    Sinv = safe_inv(Sigma)
    num = (Sinv @ ones)
    den = (ones.T @ Sinv @ ones).item()
    return (num / den).flatten()

def tangency_weights(mu: np.ndarray, rf: float, Sigma: np.ndarray) -> np.ndarray:
    """Max Sharpe weights (sum to 1)."""
    ones = np.ones((Sigma.shape[0], 1))
    Sinv = safe_inv(Sigma)
    z = Sinv @ (mu.reshape(-1, 1) - rf * ones)
    w = z / (ones.T @ z)  # normalize to sum to 1
    return w.flatten()

def frontier_curve(mu: np.ndarray, Sigma: np.ndarray, npts=151, w1_min=-0.2, w1_max=1.2):
    """Return arrays of (sigma, mean) along the 2-asset fully-invested line w2=1-w1."""
    w1 = np.linspace(w1_min, w1_max, npts)
    w2 = 1 - w1
    W = np.vstack([w1, w2]).T  # (npts x 2)
    means = W @ mu
    vars_ = np.einsum('ij,jk,ik->i', W, Sigma, W)  # row-wise quadratic form
    sigmas = np.sqrt(np.maximum(vars_, 0))
    return w1, w2, sigmas, means

# ---------- load rf ----------
# from Q1 (monthly): data/q1_rf.csv with column 'rf_monthly'
rf_path = DATA / "q1_rf.csv"
if rf_path.exists():
    rf = float(pd.read_csv(rf_path).iloc[0, 0])
else:
    # Fallback: 0.2% per month if file missing
    rf = 0.002

# ---------- load inputs (binomial params for Numerical Part) ----------
u1 = 0.10;  d1 = -0.12
u2 = 0.25;  d2 = -0.20
p  = 0.6    # same π for both assets
pi1 = pi2 = p

# means from binomial (simple returns)
mu1 = pi1 * u1 + (1 - pi1) * d1
mu2 = pi2 * u2 + (1 - pi2) * d2

# std devs from binomial: Var(R_i) = π_i(1-π_i)(u_i - d_i)^2
s1 = abs(u1 - d1) * np.sqrt(pi1 * (1 - pi1))
s2 = abs(u2 - d2) * np.sqrt(pi2 * (1 - pi2))

# store effective params so the report can show them
pd.DataFrame({"mu1":[mu1], "mu2":[mu2], "sigma1":[s1], "sigma2":[s2]}).to_csv(
    DATA / "q3_params_effective.csv", index=False
)

# ---------- main computations for two correlations ----------
rho1 = 0.5
cov12_r1 = rho1 * s1 * s2
w1_r1 = (s2 * s2 - cov12_r1) / (s1 * s1 + s2 * s2 - 2 * cov12_r1)
w2_r1 = 1 - w1_r1

rho2 = 0.0
cov12_r2 = rho2 * s1 * s2
w1_r2 = (s2 * s2 - cov12_r2) / (s1 * s1 + s2 * s2 - 2 * cov12_r2)
w2_r2 = 1 - w1_r2

pd.DataFrame({"rho1":[rho1], "rho1_w1":[w1_r1], "rho1_w2":[w2_r1], "rho2":[rho2], "rho2_w1":[w1_r2], "rho2_w2":[w2_r2]}).to_csv(
    DATA / "q3_weights.csv", index=False
)

rhos = [0.5, 0.0]
rows = []

for rho in rhos:
    # covariance matrix
    cov12 = rho * s1 * s2
    Sigma = np.array([[s1**2, cov12],
                      [cov12,  s2**2]])
    mu = np.array([mu1, mu2])

    # GMV
    w_gmv = gmv_weights(Sigma)
    mu_gmv = float(w_gmv @ mu)
    sig_gmv = float(np.sqrt(w_gmv @ Sigma @ w_gmv))

    # Tangency (max Sharpe)
    w_sr = tangency_weights(mu, rf, Sigma)
    mu_sr = float(w_sr @ mu)
    sig_sr = float(np.sqrt(w_sr @ Sigma @ w_sr))
    sharpe_sr = (mu_sr - rf) / sig_sr if sig_sr > 0 else np.nan

    rows.append({
        "rho": rho,
        "wGMV_1": w_gmv[0], "wGMV_2": w_gmv[1],
        "mu_GMV": mu_gmv,   "sigma_GMV": sig_gmv,
        "wSR_1": w_sr[0],   "wSR_2": w_sr[1],
        "mu_SR": mu_sr,     "sigma_SR": sig_sr, "sharpe_SR": sharpe_sr
    })

    # frontier plot
    w1, w2, sigmas, means = frontier_curve(mu, Sigma)
    fig, ax = plt.subplots()
    ax.plot(sigmas, means, lw=2)
    # mark GMV and Tangency
    ax.scatter([sig_gmv], [mu_gmv], marker="o")
    ax.scatter([sig_sr],  [mu_sr],  marker="*")
    ax.set_xlabel("Portfolio sigma (std dev)")
    ax.set_ylabel("Portfolio mean return")
    ax.set_title(f"Two-asset efficient frontier (rho={rho:.1f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    outpng = FIGS / f"q3_frontier_rho_{str(rho).replace('.','p')}.png"
    fig.savefig(outpng, dpi=160, bbox_inches="tight")
    plt.close(fig)

# save summary table
pd.DataFrame(rows).round(6).to_csv(DATA / "q3_portfolios.csv", index=False)

print("Saved:", DATA / "q3_params_effective.csv")
print("Saved:", DATA / "q3_weights.csv")
print("Saved:", DATA / "q3_portfolios.csv")
print("Saved:", FIGS / "q3_frontier_rho_0p5.png")
print("Saved:", FIGS / "q3_frontier_rho_0p0.png")
