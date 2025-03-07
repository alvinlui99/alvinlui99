# Standard library imports
import numpy as np
import matplotlib.pyplot as plt

# Third-party imports

# Local application/library specific imports
from config import BaseConfig
from utils import utils
from model import LGBMmodel

if __name__ == "__main__":
    dfs = utils.load_dfs_from_csv(BaseConfig.SYMBOLS, 'data/klines.csv')
    train_dfs, val_dfs, test_dfs = utils.split_dfs(dfs)

    model = LGBMmodel(BaseConfig.SYMBOLS)
    # model.train(train_dfs, val_dfs)
    # model.save_model('model/trained_models')
    model.load_model('model/trained_models')

    predictions = model.predict(test_dfs)

    for symbol in BaseConfig.SYMBOLS:
        actual = test_dfs[symbol]['Close'].pct_change(fill_method=None).shift(-1).values[:-1]
        predicted = predictions[symbol]
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        print(f"RMSE for {symbol}: {rmse}")
        
