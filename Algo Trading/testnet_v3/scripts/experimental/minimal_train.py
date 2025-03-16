"""
Minimal training script to diagnose and fix LightGBM errors.
Uses direct LightGBM training rather than the complex model class.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load a small sample of data for one symbol."""
    try:
        symbol = "BTCUSDT"
        filename = f'data/{symbol}_1h.csv'
        
        if not os.path.exists(filename):
            filename = f'data/klines_{symbol}.csv'
            if not os.path.exists(filename):
                logger.error(f"No data found for {symbol}")
                return None
        
        # Load the data
        df = pd.read_csv(filename)
        logger.info(f"Loaded {len(df)} rows from {filename}")
        
        # In your environment, we need to use more rows 
        # Make sure we have enough data for training
        if len(df) > 1000:
            # Use more data to ensure we have enough after cleaning
            df = df.iloc[-1000:].copy()
            logger.info(f"Using last 1000 rows")
        
        # Ensure we have the key columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Standardize column names if needed
        if 'Open time' in df.columns:
            logger.info(f"Converting column names")
            df = df.rename(columns={
                'Open time': 'Open_time',
                'Close time': 'Close_time',
                'Quote asset volume': 'Quote_asset_volume',
                'Number of trades': 'Number_of_trades',
                'Taker buy base asset volume': 'Taker_buy_base_asset_volume',
                'Taker buy quote asset volume': 'Taker_buy_quote_asset_volume'
            })
        
        # Check for missing columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Convert to numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for and remove NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.info(f"Dropping {nan_count} NaN values")
            df = df.dropna()
        
        logger.info(f"Data shape after cleaning: {df.shape}")
        return df
    
    except Exception as e:
        logger.exception(f"Error loading data: {str(e)}")
        return None

def create_features(df):
    """
    Create a minimal set of features for training.
    Keep it simple to avoid potential issues.
    """
    try:
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Add some basic features
        # SMA
        data['SMA5'] = data['Close'].rolling(window=5).mean()
        data['SMA10'] = data['Close'].rolling(window=10).mean()
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        
        # Price relative to SMA
        data['Close_SMA5_Ratio'] = data['Close'] / data['SMA5']
        data['Close_SMA10_Ratio'] = data['Close'] / data['SMA10']
        data['Close_SMA20_Ratio'] = data['Close'] / data['SMA20']
        
        # Volatility
        data['Volatility'] = data['High'] - data['Low']
        data['Volatility_SMA5'] = data['Volatility'].rolling(window=5).mean()
        
        # Volume
        data['Volume_SMA5'] = data['Volume'].rolling(window=5).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA5']
        
        # Drop NaN values created by rolling windows
        nan_before = data.isna().sum().sum()
        data = data.dropna()
        nan_after = data.isna().sum().sum()
        logger.info(f"Created features, dropped {nan_before - nan_after} NaN values, final shape: {data.shape}")
        
        return data
    
    except Exception as e:
        logger.exception(f"Error creating features: {str(e)}")
        return None

def train_simple_model(df):
    """
    Train a minimal LightGBM model directly.
    Uses simple splitting and minimal parameters.
    """
    try:
        # Split into features and target
        # Don't drop price columns yet - we need Close for the target
        
        # Calculate target first - next period's return
        target = df['Close'].pct_change().shift(-1).values[:-1]
        
        # Now drop price columns for features
        features = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, errors='ignore')
        features = features.iloc[:-1]  # Align with target by removing last row
        
        logger.info(f"Features shape: {features.shape}, Target shape: {target.shape}")
        logger.info(f"Feature columns: {list(features.columns)}")
        
        # Check for NaN values
        if features.isna().any().any():
            logger.warning("Features contain NaN values, dropping rows")
            valid_idx = ~features.isna().any(axis=1)
            features = features.loc[valid_idx]
            target = target[valid_idx]
            logger.info(f"After dropping NaNs - Features: {features.shape}, Target: {target.shape}")
        
        # Check for infinite values
        if np.isinf(features.values.astype(float)).any():
            logger.warning("Features contain infinite values, replacing with large values")
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)
        
        # Split into train/val
        train_size = int(len(features) * 0.8)
        X_train = features.iloc[:train_size]
        y_train = target[:train_size]
        X_val = features.iloc[train_size:]
        y_val = target[train_size:]
        
        logger.info(f"Training with {len(X_train)} rows, validating with {len(X_val)} rows")
        
        # Check that we have data
        if len(X_train) == 0 or len(X_val) == 0:
            logger.error("Not enough data for training or validation")
            return None
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Set parameters - use very simple ones
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'verbose': 0
        }
        
        # Train model
        logger.info("Training model...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=10,  # Use a small number for testing
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val']
        )
        
        logger.info("Model training successful!")
        
        # Make predictions
        preds = model.predict(X_val)
        rmse = np.sqrt(np.mean((preds - y_val) ** 2))
        logger.info(f"Validation RMSE: {rmse}")
        
        # Save the model
        os.makedirs('model/test_minimal', exist_ok=True)
        model.save_model('model/test_minimal/model.lgb')
        logger.info("Model saved to model/test_minimal/model.lgb")
        
        return model
    
    except Exception as e:
        logger.exception(f"Error training model: {str(e)}")
        return None

def main():
    """Main function for minimal model training."""
    try:
        print("Starting minimal training process...")
        
        # Load data
        df = load_data()
        if df is None or len(df) < 100:
            print("Failed to load enough data. Exiting.")
            return False
        
        # Create features
        featured_df = create_features(df)
        if featured_df is None or len(featured_df) < 50:
            print("Failed to create enough features. Exiting.")
            return False
        
        print(f"Successfully prepared data with {len(featured_df)} rows and {featured_df.shape[1]} columns")
        
        # Train model
        model = train_simple_model(featured_df)
        if model is None:
            print("Failed to train model. Exiting.")
            return False
        
        print("\n✅ Minimal training completed successfully!")
        print("This confirms LightGBM is working correctly.")
        print("Now we need to fix the data preparation in your retrain_model.py script.")
        return True
    
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 