import numpy as np
import pandas as pd

def to_monthly(df):
    # Assumes df has DatetimeIndex and 'Adj Close' column
    m = df['Adj Close'].resample('M').last().to_frame('adj_close')
    m.index = m.index.to_period('M').to_timestamp('M')
    return m

def monthly_returns(prices: pd.Series):
    return prices.pct_change().dropna()

def save_table(df: pd.DataFrame, path: str):
    df.to_csv(path, index=True)

def save_fig(fig, path: str):
    fig.savefig(path, bbox_inches='tight', dpi=160)
