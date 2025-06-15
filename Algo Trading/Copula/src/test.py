import numpy as np
from hmmlearn import hmm
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint

from collector import BinanceDataCollector
from config import Config

def exp_smoothed_return(data: pd.Series, window: int = 12, alpha: float = 0.1) -> pd.Series:
    returns = data.ewm(alpha=alpha)
    return pd.Series(returns)

if __name__ == "__main__":
    collector = BinanceDataCollector()
    coins = Config().coins
    selected_pairs = list(combinations(coins, 2))
    data = collector.get_multiple_symbols_data(coins,
                                               start_str="2023-07-01 00:00:00",
                                               end_str="2023-12-31 23:59:59")

    output = {}
    for c1, c2 in selected_pairs:
        returns1 = exp_smoothed_return(data[c1]['close'])
        returns2 = exp_smoothed_return(data[c2]['close'])
        # returns1 = data[c1]['close'].pct_change().dropna()
        # returns2 = data[c2]['close'].pct_change().dropna()
        print(returns1)
        print(returns2)
        coint_t, p_value, _ = coint(returns1, returns2)
        output[f"{c1}-{c2}"] = {
            "t": coint_t,
            "p": p_value
        }
    pd.DataFrame(output).T.to_csv("analytics/coint_test_smoothed.csv")
    # pd.DataFrame(output).T.to_csv("analytics/coint_test.csv")