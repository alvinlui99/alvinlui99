import lightgbm as lgb
import pandas as pd
import numpy as np
from utils.feature_engineering import FeatureEngineer
import logging
import os

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
        
        # Use ModelConfig's feature configuration if none provided
        if feature_config is None:
            from config import ModelConfig
            feature_config = ModelConfig.FEATURE_CONFIG
            self.logger.info("Using default feature configuration from ModelConfig")
            
        self.feature_engineer = FeatureEngineer(feature_config, logger=self.logger)
        self.logger.info(f"Initialized feature engineering with config: {feature_config}")
        
        # Added for compatibility
        self.expected_feature_counts = {}  # Will be populated when loading models
        self.compatibility_mode = False  # Allow prediction with different feature counts
    
    def train(self, klines_train: dict[str, pd.DataFrame], klines_val: dict[str, pd.DataFrame]):
        input_train, target_train = self._convert_klines(klines_train)
        input_val, target_val = self._convert_klines(klines_val)
        
        # Validate that we have enough data for training
        if input_train.size == 0 or not target_train:
            raise ValueError("No valid training data available after processing")
            
        if input_val.size == 0 or not target_val:
            raise ValueError("No valid validation data available after processing")
        
        # Verify target data shapes
        for symbol in self.symbols:
            if symbol not in target_train or len(target_train[symbol]) == 0:
                raise ValueError(f"No target data for {symbol} in training set")
            if symbol not in target_val or len(target_val[symbol]) == 0:
                raise ValueError(f"No target data for {symbol} in validation set")
                
        # Make sure we have enough rows after processing
        self.logger.info(f"Training data shape: {input_train.shape}, Validation data shape: {input_val.shape}")
        if len(input_train) < 10 or len(input_val) < 5:
            raise ValueError(f"Not enough data after processing. Need at least 10 training rows and 5 validation rows")
            
        # CRITICAL: Check for NaN values
        if np.isnan(input_train).any() or np.isnan(input_val).any():
            self.logger.warning("Input data contains NaN values - attempting to fix")
            input_train = np.nan_to_num(input_train)
            input_val = np.nan_to_num(input_val)
        
        # Store feature counts for future reference
        self.feature_count = input_train.shape[1]
        self.logger.info(f"Training with {self.feature_count} features")

        for symbol in self.symbols:
            # Check that we have target data
            if symbol not in target_train or symbol not in target_val:
                self.logger.warning(f"Missing target data for symbol {symbol}, skipping")
                continue
                
            # Prepare the dataset for LightGBM
            try:
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
                self.expected_feature_counts[symbol] = self.feature_count
                
                self.logger.info(f"Model for {symbol} trained successfully")
            except Exception as e:
                self.logger.error(f"Error training model for {symbol}: {str(e)}")
                raise ValueError(f"Training failed for {symbol}: {str(e)}")

    def predict(self, klines: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
        """
        Make predictions using the trained models.
        
        Args:
            klines: Dictionary of DataFrames containing price/volume data
            
        Returns:
            Dictionary of predictions for each symbol
        """
        self.logger.info("Starting prediction process...")
        predictions = {}

        # First check if we have any data
        if not klines:
            self.logger.error("No input data provided")
            return {}
            
        self.logger.info(f"Processing data for symbols: {list(klines.keys())}")
        
        # Process each symbol's data
        for symbol, df in klines.items():
            self.logger.info(f"\nProcessing {symbol}:")
            self.logger.info(f"Input data shape: {df.shape}")
            self.logger.info(f"Input columns: {df.columns.tolist()}")
            
            try:
                # Feature engineering
                self.logger.info("Applying feature engineering...")
                processed_data = self.feature_engineer.add_all_features(df)
                
                if processed_data.empty:
                    self.logger.warning(f"No valid data after feature engineering for {symbol}")
                    predictions[symbol] = np.array([])
                    continue
                    
                self.logger.info(f"Processed data shape: {processed_data.shape}")
                self.logger.info(f"Processed columns: {processed_data.columns.tolist()}")
                
                # Convert to numpy array
                input_data = processed_data.values
                
                # Get predictions
                if symbol in self.models:
                    try:
                        self.logger.info("Making predictions...")
                        pred = self.models[symbol].predict(input_data)
                        self.logger.info(f"Generated predictions shape: {pred.shape}")
                        self.logger.info(f"Last 5 predictions: {pred[-5:] if len(pred) > 0 else 'empty'}")
                        predictions[symbol] = pred
                    except Exception as e:
                        self.logger.error(f"Error predicting for {symbol}: {str(e)}", exc_info=True)
                        predictions[symbol] = np.array([])
                else:
                    self.logger.warning(f"No trained model found for {symbol}")
                    predictions[symbol] = np.array([])
                    
            except Exception as e:
                self.logger.error(f"Error processing data for {symbol}: {str(e)}", exc_info=True)
                predictions[symbol] = np.array([])
        
        return predictions

    def save_model(self, path: str):
        """Save the model and feature count information to disk."""
        for symbol, model in self.models.items():
            model.save_model(f'{path}/model_{symbol}.lgb')
            
        # Save feature count information
        import json
        with open(f'{path}/feature_counts.json', 'w') as f:
            json.dump(self.expected_feature_counts, f)

    def load_model(self, path: str):
        """Load the model and feature count information from disk."""
        # Load models
        for symbol in self.symbols:
            model_path = f'{path}/model_{symbol}.lgb'
            if os.path.exists(model_path):
                self.models[symbol] = lgb.Booster(model_file=model_path)
            else:
                self.logger.warning(f"Model file for {symbol} not found at {model_path}")
                
        # Try to load feature count information
        feature_counts_path = f'{path}/feature_counts.json'
        if os.path.exists(feature_counts_path):
            import json
            try:
                with open(feature_counts_path, 'r') as f:
                    self.expected_feature_counts = json.load(f)
                self.logger.info(f"Loaded feature count information: {self.expected_feature_counts}")
            except Exception as e:
                self.logger.warning(f"Failed to load feature count information: {str(e)}")
                # Extract from first model if available
                if self.models:
                    first_symbol = next(iter(self.models))
                    if hasattr(self.models[first_symbol], 'num_feature'):
                        for symbol in self.symbols:
                            if symbol in self.models:
                                self.expected_feature_counts[symbol] = self.models[symbol].num_feature()
        else:
            # Extract from models if available
            if self.models:
                for symbol in self.symbols:
                    if symbol in self.models and hasattr(self.models[symbol], 'num_feature'):
                        self.expected_feature_counts[symbol] = self.models[symbol].num_feature()

    def _convert_klines(self, klines: dict[str, pd.DataFrame]) -> tuple[np.ndarray, dict]:
        """
        Convert klines data to input and target arrays for the model.
        
        Args:
            klines: Dictionary of price data
            
        Returns:
            Tuple of input data and target data
        """
        self.logger.debug("Converting klines to model input format")
        
        # Check if this is live data from DataFetcher with Return columns
        contains_returns = any('Return' in df.columns for df in klines.values())
        if contains_returns:
            self.logger.debug("Input klines contain Return columns - this may cause feature count mismatch")
            
            # Remove Return columns for consistency with training data
            for symbol, df in klines.items():
                if 'Return' in df.columns:
                    klines[symbol] = df.drop(['Return'], axis=1)
                if 'Log_Return' in df.columns:
                    klines[symbol] = df.drop(['Log_Return'], axis=1)
        
        # First, store original Close prices for target calculation
        close_prices = {}
        for symbol, df in klines.items():
            # Allow for case-insensitive column matching
            close_col = next((col for col in df.columns if col.lower() == 'close'), None)
            if close_col:
                close_prices[symbol] = df[close_col].copy()
            else:
                self.logger.warning(f"No Close price column found for {symbol}")
        
        # Process each dataframe to add features
        processed_dfs = {}
        for symbol, df in klines.items():
            try:
                # Apply feature engineering
                processed = self.feature_engineer.add_all_features(df.copy())
                processed_dfs[symbol] = processed
            except Exception as e:
                self.logger.warning(f"Feature engineering failed for {symbol}: {str(e)}")
        
        # Handle case where feature engineering fails for all symbols
        if not processed_dfs:
            self.logger.error("Feature engineering failed for all symbols")
            return np.array([]), {}
        
        # Log warning for any symbols that failed feature engineering
        missing_symbols = set(klines.keys()) - set(processed_dfs.keys())
        if missing_symbols:
            self.logger.warning(f"Feature engineering failed for symbols: {missing_symbols}")
        
        # NaN value handling - CRITICAL CHANGE: Fill NaNs instead of dropping rows
        fixed_dfs = {}
        for symbol, df in processed_dfs.items():
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                self.logger.warning(f"Found {nan_count} NaN values in feature-engineered data for {symbol}")
                # First try forward fill, then backward fill
                fixed_df = df.ffill().bfill()
                
                # If there are still NaNs, fill with zeros
                remaining_nans = fixed_df.isna().sum().sum()
                if remaining_nans > 0:
                    self.logger.warning(f"Still have {remaining_nans} NaN values after ffill/bfill, filling with zeros")
                    fixed_df = fixed_df.fillna(0)
                
                fixed_dfs[symbol] = fixed_df
                self.logger.info(f"Successfully fixed NaN values for {symbol}: {len(fixed_df)} rows preserved")
            else:
                fixed_dfs[symbol] = df
        
        # Always check for feature count consistency
        feature_counts = {symbol: df.shape[1] for symbol, df in fixed_dfs.items()}
        unique_counts = set(feature_counts.values())
        if len(unique_counts) > 1:
            self.logger.warning(f"Inconsistent feature counts across symbols: {feature_counts}")
        
        # Track indexes to filter rows with NaNs to ensure consistency
        # No need to drop rows - we've filled NaNs above
        filtered_dfs = fixed_dfs
        
        # Prepare containers for input and target data
        all_input_data = []
        target_data = {}
        
        if not filtered_dfs:
            self.logger.error("No data available after filtering")
            return np.array([]), {}
        
        # Process feature and target data for each symbol
        for symbol, df in filtered_dfs.items():
            # Extract all features for input data
            # Exclude any index columns and timestamp columns that might be present
            feature_cols = [col for col in df.columns 
                            if col not in ['index', 'Open_time', 'Open time', 'Close_time', 'Close time'] 
                            and not pd.api.types.is_datetime64_any_dtype(df[col])
                            and not isinstance(df[col].iloc[0] if len(df) > 0 else None, pd.Timestamp)]
            
            # Log the excluded columns for debugging
            all_cols = set(df.columns)
            excluded_cols = all_cols - set(feature_cols)
            if excluded_cols:
                self.logger.debug(f"Excluded columns for {symbol}: {excluded_cols}")
            
            # Try to convert all feature columns to numeric
            features_df = df[feature_cols].copy()
            for col in features_df.columns:
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Failed to convert column {col} to numeric: {str(e)}")
                
            # Fill any remaining NaNs created during conversion
            features_df = features_df.fillna(0)
            
            # Check for infinity values
            if (features_df.values == np.inf).any() or (features_df.values == -np.inf).any():
                self.logger.warning(f"Found infinite values in {symbol} features, replacing with large values")
                features_df = features_df.replace([np.inf, -np.inf], [1e9, -1e9])
            
            all_input_data.append(features_df.values)
            
            # Calculate target using the stored close data
            if symbol in close_prices:
                # Get Close prices for this symbol
                symbol_close = close_prices[symbol]
                
                # If we have an index column in the filtered dataframe, use it to align
                if 'index' in df.columns:
                    try:
                        # Extract indexes to filter
                        indexes = df['index'].values
                        filtered_close = symbol_close.iloc[indexes]
                    except Exception as e:
                        self.logger.warning(f"Error filtering close prices using index: {str(e)}")
                        # Fallback to using Close from the filtered dataframe
                        close_col = next((col for col in df.columns if col.lower() == 'close'), None)
                        filtered_close = df[close_col] if close_col else None
                else:
                    # If we don't have an index column, assume rows are already aligned
                    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
                    filtered_close = df[close_col] if close_col else None
                
                if filtered_close is not None:
                    # Calculate percent change for targets
                    try:
                        target = filtered_close.pct_change(fill_method=None).shift(-1).values
                        # Remove the last value which will be NaN
                        target_data[symbol] = target[:-1]
                        self.logger.debug(f"Target data shape for {symbol}: {target_data[symbol].shape}")
                    except Exception as e:
                        self.logger.warning(f"Error calculating target for {symbol}: {str(e)}")
                else:
                    self.logger.warning(f"Could not calculate target for {symbol}, no close price data")
            else:
                self.logger.warning(f"No original close price data for {symbol}")
        
        # Debug data shapes
        for i, data in enumerate(all_input_data):
            self.logger.debug(f"Input data shape for symbol {i}: {data.shape}")
        
        # Ensure all feature arrays have the same shape
        if not all_input_data:
            self.logger.error("No input data available")
            return np.array([]), {}
        
        # Get the minimum number of rows across all symbols
        min_rows = min(data.shape[0] for data in all_input_data)
        self.logger.debug(f"Min rows across all symbols: {min_rows}")
        
        # Make sure min_rows is greater than 0
        if min_rows <= 0:
            self.logger.error("No valid rows found in input data")
            return np.array([]), {}
        
        # Truncate all arrays to the same length
        adjusted_inputs = [data[:min_rows] for data in all_input_data]
        
        # Ensure target data is adjusted in the same way
        adjusted_targets = {}
        for symbol, targets in target_data.items():
            # Ensure we don't have an indexing error
            target_len = min(min_rows-1, len(targets))
            if target_len > 0:
                adjusted_targets[symbol] = targets[:target_len]
            else:
                self.logger.warning(f"No valid target data for {symbol} after adjustment")
        
        # Concatenate all input data
        if adjusted_inputs:
            try:
                # Verify input data doesn't have NaN or Inf values
                for i, data in enumerate(adjusted_inputs):
                    if np.isnan(data).any():
                        self.logger.warning(f"Input data for symbol {i} still contains NaN values, filling with 0")
                        adjusted_inputs[i] = np.nan_to_num(data, nan=0.0)
                    if np.isinf(data).any():
                        self.logger.warning(f"Input data for symbol {i} contains Inf values, replacing with large values")
                        adjusted_inputs[i] = np.nan_to_num(data, posinf=1e9, neginf=-1e9)
                
                input_data = np.concatenate(adjusted_inputs, axis=1)
                
                # Final verification for NaN and Inf values
                if np.isnan(input_data).any():
                    self.logger.warning("Concatenated input data contains NaN values, filling with 0")
                    input_data = np.nan_to_num(input_data, nan=0.0)
                if np.isinf(input_data).any():
                    self.logger.warning("Concatenated input data contains Inf values, replacing with large values")
                    input_data = np.nan_to_num(input_data, posinf=1e9, neginf=-1e9)
                    
                # Make sure we have at least one row
                if input_data.shape[0] <= 1:
                    self.logger.error("Insufficient data rows after processing")
                    return np.array([]), {}
                    
                # Final check - we need to remove the last row because there's no target for it
                final_input = input_data[:-1]
                self.logger.debug(f"Final input data shape: {final_input.shape}")
                
                # Safety check to make sure we don't return an empty array
                if final_input.size == 0:
                    self.logger.error("Empty input data after processing")
                    return np.array([[0]]), {}  # Return a dummy array
                    
                return final_input, adjusted_targets
            except ValueError as e:
                self.logger.error(f"Error concatenating input data: {str(e)}")
                self.logger.debug(f"Input data shapes: {[data.shape for data in adjusted_inputs]}")
                # This should not happen with the improved alignment approach
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
