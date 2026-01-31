import numpy as np
import pandas as pd


def rolling_var(returns: pd.Series, window: int = 60, alpha: float = 0.95):
    """
    Historical 1-day VAR (absolute, not %) using |window| trading days.
    Returns a Series aligned with input (NaN until window).
    """
    def _var(x):
        return np.quantile(x, 1 - alpha)
    return returns.rolling(window).apply(_var, raw=True)
