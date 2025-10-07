# Question 1 calculations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import monthly_returns, save_table, save_fig

DATA = Path(__file__).resolve().parents[1] / "data"
FIGS = Path(__file__).resolve().parents[1] / "figures"
FIGS.mkdir(exist_ok=True)

spy = pd.read_csv(DATA / "SPY_monthly_2012_2022.csv", parse_dates=['Date'], index_col='Date')
r = monthly_returns(spy['adj_close'])

# Estimate binomial parameters π, u, d using sign and conditional means
pi_hat = (r > 0).mean()
u_hat  = r[r > 0].mean()
d_hat  = r[r <= 0].mean()

est = pd.DataFrame({"pi":[pi_hat],"u":[u_hat],"d":[d_hat]})
save_table(est, DATA / "q1_binomial_params.csv")

# Plot monthly return histogram (no seaborn)
fig, ax = plt.subplots()
ax.hist(r.values, bins=30)
ax.set_title("SPY Monthly Returns (2012–2022)")
ax.set_xlabel("Return")
ax.set_ylabel("Freq")
save_fig(fig, FIGS / "q1_spy_return_hist.png")

print(est)
