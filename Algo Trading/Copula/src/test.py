import numpy as np
from hmmlearn import hmm
import pandas as pd

from collector import BinanceDataCollector

if __name__ == "__main__":
    collector = BinanceDataCollector()
    data = collector.get_multiple_symbols_data(["BTCUSDT", "ETHUSDT"],
                                               start_str="2023-07-01 00:00:00",
                                               end_str="2023-12-31 00:00:00")
    models = []

    for symbol in data:
        returns = data[symbol]['close'].pct_change()
        volatility = returns.rolling(window=10*24).std()
        
        # Create DataFrame and handle NaN values
        df = pd.DataFrame({
            'returns': returns,
            'volatility': volatility
        })
        df = df.dropna()
        features = df.values
        model = hmm.GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=1000
        )
        model.fit(features)
        models.append(model)
        print(symbol)
        print(model.means_)
        print(model.covars_)
        print(model.transmat_)
        print(model.startprob_)