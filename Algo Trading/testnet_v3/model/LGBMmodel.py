import lightgbm as lgb
import pandas as pd
import numpy as np
from utils.feature_engineering import FeatureEngineer
import logging

class LGBMmodel:
    def __init__(self, symbols: list[str], feature_config=None, logger=None):
        """
        Initialize the LightGBM model.
        
        Args:
            symbols: List of trading symbols
            feature_config: Optional configuration for feature engineering
            logger: Optional logger instance
        """
        self.symbols = symbols
        self.models = {}
        self.logger = logger or logging.getLogger(__name__)
        self.feature_engineer = FeatureEngineer(feature_config, logger=self.logger)
    
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
        """
        Convert klines data to input and target arrays for the model.
        
        Args:
            klines: Dictionary of price data
            
        Returns:
            Tuple of input data and target data
        """
        self.logger.debug("Converting klines to model input format")
        
        # First, store original Close prices for target calculation
        close_prices = {}
        for symbol, df in klines.items():
            if 'Close' in df.columns:
                close_prices[symbol] = df['Close'].copy()
        
        # Process each dataframe to add features
        processed_dfs = {}
        for symbol in self.symbols:
            if symbol not in klines:
                error_msg = f"Symbol {symbol} not found in provided klines data"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
            df = klines[symbol].copy()
            
            # Ensure data types are correct for TA-Lib functions
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Use the feature engineer to add features
            try:
                df = self.feature_engineer.add_all_features(df)
                # Keep track of the index
                if hasattr(df, 'index'):
                    df = df.reset_index(drop=False)
                processed_dfs[symbol] = df
            except Exception as e:
                self.logger.warning(f"Error during feature engineering for {symbol}: {str(e)}")
                # Skip this symbol if feature engineering fails completely
        
        # Early exit if no data was processed
        if not processed_dfs:
            self.logger.error("No data could be processed for any symbol")
            return np.array([]), {}
        
        # Find common rows that are valid across all dataframes (no NaNs)
        common_valid_rows = None
        
        for symbol, df in processed_dfs.items():
            # Get mask of non-NaN rows
            valid_rows = ~df.isnull().any(axis=1)
            
            if common_valid_rows is None:
                common_valid_rows = valid_rows
            else:
                # Keep only rows that are valid in all dataframes
                common_valid_rows = common_valid_rows & valid_rows
        
        # Filter dataframes to include only rows with no NaNs in any dataframe
        filtered_dfs = {}
        for symbol, df in processed_dfs.items():
            filtered_dfs[symbol] = df.loc[common_valid_rows].copy()
            self.logger.debug(f"Filtered {symbol} dataframe from {len(df)} to {len(filtered_dfs[symbol])} rows")
        
        # Prepare input and target data
        all_input_data = []
        target_data = {}
        
        for symbol, df in filtered_dfs.items():
            # Extract all features for input data
            # Exclude any index columns that might be present
            feature_cols = [col for col in df.columns if col not in ['index', 'Open_time', 'Open time']]
            features_df = df[feature_cols]
            all_input_data.append(features_df.values)
            
            # Calculate target using the stored close data
            if symbol in close_prices:
                # Filter original close prices to match the filtered rows
                if hasattr(df, 'index') and 'index' in df.columns:
                    filtered_close = close_prices[symbol].iloc[df['index']]
                else:
                    # Fallback to using the close prices from the filtered dataframe
                    filtered_close = df['Close'] if 'Close' in df.columns else None
                
                if filtered_close is not None:
                    # Calculate percent change for targets
                    target = filtered_close.pct_change(fill_method=None).shift(-1).values
                    # Remove the last value which will be NaN
                    target_data[symbol] = target[:-1]
                else:
                    self.logger.warning(f"Could not calculate target for {symbol}, no close price data")
            else:
                self.logger.warning(f"No original close price data for {symbol}")
        
        # Ensure all feature arrays have the same shape
        min_rows = min(data.shape[0] for data in all_input_data)
        adjusted_inputs = [data[:min_rows] for data in all_input_data]
        adjusted_targets = {symbol: targets[:min_rows-1] for symbol, targets in target_data.items()}
        
        # Concatenate all input data
        if adjusted_inputs:
            try:
                input_data = np.concatenate(adjusted_inputs, axis=1)
                self.logger.debug(f"Final input data shape: {input_data[:-1].shape}")
                return input_data[:-1], adjusted_targets
            except ValueError as e:
                self.logger.error(f"Error concatenating input data: {str(e)}")
                self.logger.debug(f"Input data shapes: {[data.shape for data in adjusted_inputs]}")
                # This should not happen with the new alignment approach
                raise
        else:
            self.logger.warning("No input data found after processing")
            return np.array([]), {}
    
    def update_feature_config(self, new_config):
        """
        Update the feature engineering configuration.
        
        Args:
            new_config: New configuration dictionary
        """
        self.feature_engineer.update_config(new_config)
