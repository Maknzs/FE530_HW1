# Fetch SPY (2012–2022) and SHY (2021–2022) monthly adjusted closes
import pandas as pd
import yfinance as yf
from pathlib import Path
from utils import to_monthly, save_table

out_dir = Path(__file__).resolve().parents[1] / "data"
out_dir.mkdir(exist_ok=True)

spy = yf.download('SPY', start='2012-01-01', end='2023-01-01', auto_adjust=False, progress=False)
shy = yf.download('SHY', start='2021-01-01', end='2023-01-01', auto_adjust=False, progress=False)

spy_m = to_monthly(spy)
shy_m = to_monthly(shy)

save_table(spy_m, out_dir / "SPY_monthly_2012_2022.csv")
save_table(shy_m, out_dir / "SHY_monthly_2021_2022.csv")

print('Saved to data/SPY_monthly_2012_2022.csv and data/SHY_monthly_2021_2022.csv')
