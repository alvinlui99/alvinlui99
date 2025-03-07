import lightgbm as lgb
import pandas as pd
import numpy as np
class LGBMmodel:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.models = {}
    
    def train(self, klines_train: dict[str, pd.DataFrame], klines_val: dict[str, pd.DataFrame]):
        input_train, target_train = self._convert_klines(klines_train)
        input_val, target_val = self._convert_klines(klines_val)        

        for symbol in self.symbols:
            # Prepare the dataset for LightGBM
            dtrain = lgb.Dataset(input_train, label=target_train[symbol])
            dval = lgb.Dataset(input_val, label=target_val[symbol], reference=dtrain)
            
            # Set parameters for LightGBM
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.1,
                'num_leaves': 31,
                'verbose': -1
            }
            
            # Train the model
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=100,
                valid_sets=[dtrain, dval],
                valid_names=['train', 'val']
            )
            
            # Store the trained model
            self.models[symbol] = model

    def predict(self, klines: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
        input_data, _ = self._convert_klines(klines)
        predictions = {}

        for symbol in self.symbols:
            if symbol in self.models:
                model = self.models[symbol]
                # Make predictions using the model
                predictions[symbol] = model.predict(input_data)
            else:
                raise ValueError(f"Model for symbol {symbol} not found. Please train the model first.")
        
        return predictions

    def save_model(self, path: str):
        for symbol, model in self.models.items():
            model.save_model(f'{path}/model_{symbol}.lgb')

    def load_model(self, path: str):
        for symbol in self.symbols:
            self.models[symbol] = lgb.Booster(model_file=f'{path}/model_{symbol}.lgb')
    
    def _convert_klines(self, klines: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
        all_input_data = []
        target_data = {}
        
        for symbol in self.symbols:
            df = klines[symbol]
            df = self._add_features(df)
            all_input_data.append(df)

            # Shift the 'Close' column to use the next day's closing price as the target
            target = df['Close'].pct_change(fill_method=None).shift(-1).values
            target_data[symbol] = target[:-1]  # Exclude the last element which will be NaN after shift
        
        input_data = np.concatenate(all_input_data, axis=1)
        
        return input_data[:-1], target_data

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
