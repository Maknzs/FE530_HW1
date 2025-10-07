# Question 4 helpers; simple prints only
import numpy as np
import pandas as pd
from pathlib import Path
from utils import save_table

DATA = Path(__file__).resolve().parents[1] / "data"

def forward_price(S0, RF):
    return S0 * (1 + RF)

def forward_payoff(S1, F01):
    return S1 - F01

cases = [{"S0":100,"RF":0.04,"F":v} for v in (104,105,103)]
out=[]
for c in cases:
    fair = forward_price(c["S0"], c["RF"])
    mispricing = c["F"] - fair
    out.append({"F":c["F"], "fair":fair, "mispricing":mispricing,
                "direction":"short forward (if positive, overpriced) / long forward (if negative, underpriced)"})
pd.DataFrame(out).to_csv(DATA/"q4_arbitrage_setup.csv", index=False)

print(pd.DataFrame(out))
