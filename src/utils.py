import numpy as np
import pandas as pd

def to_monthly(df):
    # Works whether df['Adj Close'] is a Series or a single-column DataFrame
    s = df['Adj Close']
    if isinstance(s, pd.DataFrame):
        # squeeze to a Series if it’s a single column; else pick the first column explicitly
        if s.shape[1] == 1:
            s = s.squeeze("columns")
        else:
            # if MultiIndex (e.g., ('Adj Close','SPY')), select the first column
            s = s.iloc[:, 0]

    # Use ME (month-end) per pandas deprecation notice
    m = s.resample('ME').last().to_frame(name='adj_close')
    # Ensure month-end timestamps
    m.index = m.index.to_period('M').to_timestamp('M')
    return m

def monthly_returns(prices: pd.Series):
    return prices.pct_change().dropna()

def save_table(df: pd.DataFrame, path: str):
    df.to_csv(path, index=True)

def save_fig(fig, path: str):
    fig.savefig(path, bbox_inches='tight', dpi=160)
