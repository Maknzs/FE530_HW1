# Question 1 calculations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import monthly_returns, save_table, save_fig

DATA = Path(__file__).resolve().parents[1] / "data"
FIGS = Path(__file__).resolve().parents[1] / "figures"
FIGS.mkdir(exist_ok=True)

spy = pd.read_csv(
    DATA / "SPY_monthly_2012_2022.csv", parse_dates=["Date"], index_col="Date"
)
r = monthly_returns(spy["adj_close"])

# --- Q1.2: Estimate binomial parameters π, u, d using sign and conditional means
pi_hat = (r > 0).mean()
u_hat = r[r > 0].mean()
d_hat = r[r <= 0].mean()

est = pd.DataFrame({"pi": [pi_hat], "u": [u_hat], "d": [d_hat]})
save_table(est, DATA / "q1_binomial_params.csv")

# Plot monthly return histogram (for fun)
fig, ax = plt.subplots()
ax.hist(r.values, bins=30)
ax.set_title("SPY Monthly Returns (2012–2022)")
ax.set_xlabel("Return")
ax.set_ylabel("Freq")
save_fig(fig, FIGS / "q1_spy_return_hist.png")

shy = pd.read_csv(
    DATA / "SHY_monthly_2021_2022.csv", parse_dates=['Date'], index_col='Date'
)
shy_r = monthly_returns(shy["adj_close"])

# --- Q1.3: Estimate rf
rf_series = monthly_returns(shy['adj_close'])
rf_hat = rf_series.mean()  # simple monthly mean
pd.DataFrame({"rf":[rf_hat]}).to_csv(DATA/"q1_rf.csv", index=False)

# --- Q1.4: no arb condition satisfied?
no_arb = (d_hat < rf_hat) and (rf_hat < u_hat)
pd.DataFrame({
    "d":[d_hat],
    "rf":[rf_hat],
    "u":[u_hat],
    "no_arbitrage":[no_arb]
}).to_csv(DATA/"q1_no_arb.csv", index=False)

# --- Q1.5: Minimize Portfolio Variance on $100

# sample variance of the risky monthly return (ddof=1)
sigma2_hat = r.var()
shy_sigma2_hat = shy_r.var()


w_grid = np.linspace(0.0, 1.0, 101)
var_V = (100.0 ** 2) * (w_grid ** 2) * sigma2_hat

# save data
var_df = pd.DataFrame({"w": w_grid, "var_V_next": var_V})
var_df.to_csv(DATA / "q1_var_curve.csv", index=False)

# plot
fig, ax = plt.subplots()
ax.plot(w_grid, var_V)
ax.set_xlabel("w (weight in risky asset)")
ax.set_ylabel(r"Var[$V_{t+1}$]")
ax.set_title("One-month portfolio variance vs. risky weight w")
ax.grid(True, alpha=0.3)
fig.savefig(FIGS / "q1_var_vs_w.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# --- Q1.6: allocation (x,y) to target E[V_{t+1}]=102 with V0=100 ---

# risky expected monthly return from binomial params
mu_hat = pi_hat * u_hat + (1 - pi_hat) * d_hat

target_gross = 1.02  # 102 / 100
den = (mu_hat - rf_hat)

if abs(den) < 1e-12:
    w = float('nan'); x = float('nan'); y = float('nan')
    regime = "mu ≈ r_f: target not identifiable (only 1+r_f achievable)"
else:
    w = (target_gross - (1 + rf_hat)) / den  # weight in risky
    x = 100 * w
    y = 100 - x
    regime = ("levered long risky" if w > 1
              else ("short risky / lend" if w < 0
                    else "long-only mix"))

pd.DataFrame({
    "pi":[pi_hat], "u":[u_hat], "d":[d_hat],
    "mu":[mu_hat], "rf":[rf_hat],
    "w":[w], "x":[x], "y":[y],
    "regime":[regime]
}).round(4).to_csv(DATA / "q1_alloc_102.csv", index=False)

# --- Q1.7: One-step European call pricing (replication and DCF) ---

import pandas as pd

S0 = B0 = 100 # assumed
K = 101

# State payoffs
Cu = max(S0 * (1 + u_hat) - K, 0.0)
Cd = max(S0 * (1 + d_hat) - K, 0.0)

den_ud = ((1 + u_hat) - (1 + d_hat))
if abs(den_ud) < 1e-12:
    pi_star = float('nan')
    x_est = float('nan')
    y_est = float('nan')
    C0_dcf = float('nan')
    C0_rep = float('nan')
    rep_up = float('nan')
    rep_down = float('nan')
else:
    x_rep = (Cu - Cd) / (S0 * (u_hat - d_hat))
    y_rep = (Cu - Cd - (Cu / (S0 * (u_hat - d_hat))) * S0 * (1 + u_hat)) / (B0 * (1+rf_hat))
    C0_rep = x_rep * S0 + y_rep * B0
    rep_up   = x_rep * S0 * (1 + u_hat) + B0 * (1 + rf_hat) - Cu
    rep_down = x_rep * S0 * (1 + d_hat) + B0 * (1 + rf_hat) - Cd
    up_down = rep_up.round(4) == rep_down.round(4)
    pi_star = (rf_hat - d_hat) / (u_hat - d_hat)
    C0_dcf = (pi_star * Cu + (1 - pi_star) * Cd) / (1 + rf_hat)
    rep_vs_dcf = C0_dcf.round(4) == C0_rep.round(4)
    
valid_pi = (0.0 <= pi_star <= 1.0) if pd.notna(pi_star) else False

out = pd.DataFrame({
    "S0":[S0], "K":[K],
    "u":[u_hat], "d":[d_hat],
    "rf":[rf_hat], "pi_star":[pi_star], "valid_pi":[valid_pi],
    "Cu":[Cu], "Cd":[Cd],
    "x_rep":[x_rep], "y_rep":[y_rep], 
    "B0":[B0], "C0_dcf":[C0_dcf],
    "C0_rep":[C0_rep],
    "Vu":[rep_up],
    "Vd":[rep_down],
    "Vu=Vd":[up_down],
    "C0_rep=C0_dcf":[rep_vs_dcf]
}).round(4)

out.to_csv(DATA / "q1_option_pricing.csv", index=False)
print(out)