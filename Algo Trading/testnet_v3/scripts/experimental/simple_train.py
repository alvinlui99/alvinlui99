import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import pickle
from datetime import datetime
from utils.feature_engineering import FeatureEngineer
from config import BaseConfig

# Set up logging with clear console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(symbols):
    """Load data files directly"""
    dfs = {}
    for symbol in symbols:
        try:
            # Try both file formats
            live_file = f'data/{symbol}_1h.csv'
            klines_file = f'data/klines_{symbol}.csv'
            
            if os.path.exists(live_file):
                df = pd.read_csv(live_file)
                logger.info(f"Loaded live data for {symbol}: {len(df)} rows")
                # Make sure we use only the latest 100 rows for consistency
                if len(df) > 100:
                    df = df.iloc[-100:].copy()
                dfs[symbol] = df
            elif os.path.exists(klines_file):
                df = pd.read_csv(klines_file)
                logger.info(f"Loaded klines data for {symbol}: {len(df)} rows")
                # Make sure we use only the latest 100 rows for consistency
                if len(df) > 100:
                    df = df.iloc[-100:].copy()
                dfs[symbol] = df
            else:
                logger.warning(f"No data file found for {symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
    
    return dfs

def prepare_data(dfs):
    """Convert raw data to features and targets"""
    # Create feature engineer
    feature_engineer = FeatureEngineer()
    
    X_train = []
    y_train = {}
    
    for symbol, df in dfs.items():
        try:
            # Convert columns to appropriate format if needed
            if 'Open time' in df.columns:
                # Convert klines format column names
                df = df.rename(columns={
                    'Open time': 'Open_time', 
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Close time': 'Close_time',
                    'Quote asset volume': 'Quote_asset_volume',
                    'Number of trades': 'Number_of_trades',
                    'Taker buy base asset volume': 'Taker_buy_base_asset_volume',
                    'Taker buy quote asset volume': 'Taker_buy_quote_asset_volume'
                })
            
            # Make sure price columns are numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                col_match = next((c for c in df.columns if c.lower() == col.lower()), None)
                if col_match:
                    df[col_match] = pd.to_numeric(df[col_match], errors='coerce')
            
            # Store close prices for target calculation
            close_col = next((col for col in df.columns if col.lower() == 'close'), None)
            if close_col:
                close_prices = df[close_col].copy()
            else:
                logger.error(f"No Close column found for {symbol}")
                continue
            
            # Make sure index is reset (critical for feature engineering)
            df = df.reset_index(drop=True)
            
            # Add features
            features_df = feature_engineer.add_all_features(df)
            
            # Handle NaN values
            nan_count = features_df.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in features for {symbol}")
                # Fill forward, backward, then zeros
                features_df = features_df.ffill().bfill().fillna(0)
                logger.info(f"Successfully filled NaN values for {symbol}")
            
            # Remove non-numeric columns
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < len(features_df.columns):
                logger.warning(f"Dropping {len(features_df.columns) - len(numeric_cols)} non-numeric columns")
                features_df = features_df[numeric_cols]
            
            # Ensure no infinity values
            if (features_df.values == np.inf).any() or (features_df.values == -np.inf).any():
                logger.warning(f"Found infinite values in {symbol} features")
                features_df = features_df.replace([np.inf, -np.inf], [1e9, -1e9])
            
            # Calculate target - next period's return
            # Use the original close prices to calculate returns
            target = close_prices.pct_change().shift(-1)
            # Drop the last row because we don't have a target for it
            target = target.iloc[:-1].values
            
            # Drop the last row of features to align with target
            features = features_df.iloc[:-1].values
            
            # Print detailed information about the data
            logger.info(f"Features shape for {symbol}: {features.shape}, Target shape: {len(target)}")
            logger.info(f"Feature columns for {symbol} ({len(features_df.columns)}): {list(features_df.columns)[:5]}...")
            
            # Make sure we have enough data and no NaN values
            if len(target) > 0 and features.shape[0] > 0:
                # Final check for NaN
                if np.isnan(features).any():
                    logger.warning(f"Features for {symbol} still contain NaN values - filling with zeros")
                    features = np.nan_to_num(features, nan=0.0)
                
                # Make sure features and target have the same number of rows
                min_rows = min(features.shape[0], len(target))
                features = features[:min_rows]
                target = target[:min_rows]
                
                # Add to training data
                logger.info(f"Adding {min_rows} rows for {symbol} to training data")
                X_train.append(features)
                y_train[symbol] = target
            else:
                logger.warning(f"Not enough data for {symbol} after processing - features: {features.shape}, target: {len(target)}")
                
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Concatenate features from all symbols
    if X_train and all(x.shape[0] > 0 for x in X_train):
        # Find minimum number of rows
        min_rows = min(x.shape[0] for x in X_train)
        logger.info(f"Using {min_rows} rows for training (minimum across all symbols)")
        
        # Truncate to same length
        X_train = [x[:min_rows] for x in X_train]
        y_train = {s: y[:min_rows] for s, y in y_train.items()}
        
        # Concatenate features horizontally
        try:
            X_combined = np.concatenate(X_train, axis=1)
            logger.info(f"Combined features shape: {X_combined.shape}")
            
            # Check for NaN values one more time
            if np.isnan(X_combined).any():
                logger.warning("Combined features contain NaN values - filling with zeros")
                X_combined = np.nan_to_num(X_combined, nan=0.0)
            
            return X_combined, y_train
        except Exception as e:
            logger.error(f"Error concatenating features: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    else:
        logger.error("No valid data for training")
        return None, None

def train_model(X, y, symbol):
    """Train a simple LightGBM model for one symbol"""
    try:
        logger.info(f"Training model for {symbol} with {X.shape[0]} rows and {X.shape[1]} features")
        
        # Final verification of data
        if np.isnan(X).any():
            logger.warning("Training features contain NaN values - filling with zeros")
            X = np.nan_to_num(X, nan=0.0)
        if np.isnan(y).any():
            logger.warning("Training target contains NaN values - filling with zeros")
            y = np.nan_to_num(y, nan=0.0)
        
        # Create LightGBM datasets
        lgb_train = lgb.Dataset(X, label=y)
        
        # Set parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'verbose': -1
        }
        
        # Train the model
        logger.info(f"Starting LightGBM training for {symbol}")
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100
        )
        
        logger.info(f"Model for {symbol} trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def save_model(model, symbol, output_dir):
    """Save the trained model to disk"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'model_{symbol}.lgb')
        model.save_model(model_path)
        logger.info(f"Model for {symbol} saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model for {symbol}: {str(e)}")
        return False

def main():
    print("\n===== SIMPLE LGBM MODEL TRAINING =====\n")
    logger.info("Starting simple model training")
    
    # Use just 2 symbols for testing
    symbols = ["BTCUSDT", "ETHUSDT"]
    logger.info(f"Training with symbols: {symbols}")
    
    try:
        # Load data
        dfs = load_data(symbols)
        if not dfs:
            logger.error("No data loaded")
            print("\n❌ No data loaded for training")
            return
        
        # Print quick summary of loaded data
        print(f"\nLoaded data for {len(dfs)} symbols:")
        for symbol, df in dfs.items():
            print(f"- {symbol}: {len(df)} rows, columns: {list(df.columns)[:5]}...")
            
        # Prepare data for training
        X, y = prepare_data(dfs)
        if X is None or y is None:
            logger.error("Failed to prepare data")
            print("\n❌ Failed to prepare data for training")
            return
            
        # Make sure we have features and targets
        if X.shape[0] == 0 or not y:
            logger.error("No data available for training")
            print("\n❌ No data available after preprocessing")
            return
            
        # Train models for each symbol
        models = {}
        for symbol, target in y.items():
            try:
                model = train_model(X, target, symbol)
                if model:
                    models[symbol] = model
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {str(e)}")
        
        if not models:
            logger.error("No models were successfully trained")
            print("\n❌ No models were successfully trained")
            return
            
        # Save models
        output_dir = os.path.join(BaseConfig.MODEL_DIR, "simple_models")
        success = True
        for symbol, model in models.items():
            if not save_model(model, symbol, output_dir):
                success = False
                
        # Save symbols list for reference
        with open(os.path.join(output_dir, 'trained_symbols.txt'), 'w') as f:
            f.write('\n'.join(models.keys()))
            
        if success:
            logger.info(f"Training completed successfully for {len(models)} symbols")
            print(f"\n✅ Models trained and saved to {output_dir}")
        else:
            logger.warning(f"Training completed with some errors")
            print(f"\n⚠️ Models trained but there were some errors during saving")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 