from sklearn.preprocessing import StandardScaler
import talib
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import BaseConfig, ModelConfig, FeatureConfig, TradingConfig
import os
import joblib
import logging

logger = logging.getLogger(__name__)

class FeaturePreprocessor:
    def __init__(self):
        self.is_configured = False
        self.is_fitted = False

    def configure(self) -> None:
        self.symbols = TradingConfig.SYMBOLS
        self.features = FeatureConfig.FEATURE_NAMES
        self.scaler_path = ModelConfig.SCALER_PATH
        self.is_configured = True

    def fit(
        self,
        dataset: dict[str, dict[str, pd.DataFrame]]
    ) -> None:
        """
        Split X y, then fit X
        """
        if not self.is_configured:
            raise ValueError("Preprocessor is not configured. Call configure() first.")
        train_X_with_features = self.prepare_features_for_symbols(dataset['train_data_X'])

        self.scaler = StandardScaler()
        self.scaler.fit(train_X_with_features)
        self.is_fitted = True
        self.save_scaler()

    def transform(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted. Call fit() first.")
        return {symbol: self.scaler.transform(symbol_data) for symbol, symbol_data in data.items()}

    def add_features(
            self,
            dataset: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        feature_data = {}
        for symbol, symbol_data in data.items():
            feature_data[symbol] = self.engineer_features(symbol_data, ModelConfig.PRICE_COLUMN)
        return feature_data
    
    def engineer_features(
            self,
            data: pd.DataFrame,
            price_column: str
    ) -> pd.DataFrame:
        """Engineer features from price data"""
        features = pd.DataFrame(index=data.index)
        
        # Price and returns
        features['return'] = data[price_column].pct_change()
        features['close'] = data[price_column]
        features['high'] = data['high']
        features['low'] = data['low']
        features['open'] = data['open']
        features['volume'] = data['volume']
        
        # Technical indicators
        # Volatility
        features['volatility'] = features['return'].rolling(
            window=FeatureConfig.LOOKBACK_PERIOD).std()
        
        # Momentum
        features['momentum'] = features['return'].rolling(
            window=FeatureConfig.LOOKBACK_PERIOD).mean()
        
        # Price relative to SMA
        sma = features['close'].rolling(window=FeatureConfig.LOOKBACK_PERIOD).mean()
        features['price_to_sma'] = features['close'] / sma
        features['price_std'] = features['close'].rolling(
            window=FeatureConfig.LOOKBACK_PERIOD).std()
        
        # RSI
        delta = features['return']
        gain = (delta.where(delta > 0, 0)).rolling(
            window=FeatureConfig.TECHNICAL_INDICATORS['RSI_PERIOD']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(
            window=FeatureConfig.TECHNICAL_INDICATORS['RSI_PERIOD']).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = features['close'].ewm(
            span=FeatureConfig.TECHNICAL_INDICATORS['MACD_FAST'], adjust=False).mean()
        exp2 = features['close'].ewm(
            span=FeatureConfig.TECHNICAL_INDICATORS['MACD_SLOW'], adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(
            span=FeatureConfig.TECHNICAL_INDICATORS['MACD_SIGNAL'], adjust=False).mean()
        
        # Bollinger Bands
        bb_sma = features['close'].rolling(
            window=FeatureConfig.TECHNICAL_INDICATORS['BB_PERIOD']).mean()
        bb_std = features['close'].rolling(
            window=FeatureConfig.TECHNICAL_INDICATORS['BB_PERIOD']).std()
        bb_upper = bb_sma + (FeatureConfig.TECHNICAL_INDICATORS['BB_STD'] * bb_std)
        bb_lower = bb_sma - (FeatureConfig.TECHNICAL_INDICATORS['BB_STD'] * bb_std)
        features['bb_position'] = (features['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Statistical moments
        features['skewness'] = features['return'].rolling(
            window=FeatureConfig.LOOKBACK_PERIOD).skew()
        features['kurtosis'] = features['return'].rolling(
            window=FeatureConfig.LOOKBACK_PERIOD).kurt()
        
        # Forward fill only and drop initial NaN rows
        features = features.ffill().dropna()
        
        return features

    def load_historical_data(
            self,
            symbols: List[str],
            start_date: str,
            end_date: str   
    ) -> Dict[str, pd.DataFrame]:
        symbol_data = {}
        all_timestamps = set()
        first_symbol = True
        
        for symbol in symbols:
            # Read and process data in one step
            df = (pd.read_csv(f"{BaseConfig.DATA_PATH}/{symbol}.csv")
                .assign(datetime=lambda x: pd.to_datetime(x['index']))
                .set_index('datetime')
                .loc[start_date:end_date])
            
            # Create price DataFrame efficiently
            symbol_data[symbol] = pd.DataFrame(
                {'price': df['Close'],
                'high': df['High'],
                'low': df['Low'],
                'open': df['Open'],
                'volume': df['Volume']},
                index=df.index
            )
            
            # Track common timestamps
            if first_symbol:
                all_timestamps = set(df.index)
                first_symbol = False
            else:
                all_timestamps = all_timestamps.intersection(set(df.index))
        
        # Filter each DataFrame to include only common timestamps
        common_timestamps = sorted(list(all_timestamps))
        for symbol in symbols:
            symbol_data[symbol] = symbol_data[symbol].loc[common_timestamps]
        
        return symbol_data

    def save_scaler(
            self,
            scaler_path: str = None
    ) -> None:
        """Save scaler to file"""
        if scaler_path is None:
            scaler_path = self.scaler_path
        try:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(self.scaler, scaler_path)
            logger.info("Feature scaler saved successfully")
        except Exception as e:
            print(f"scaler_path: {scaler_path}")
            logger.error(f"Error saving scaler: {e}")
            raise

    def load_scaler(
            self,
            scaler_path: str = None
    ) -> None:
        """Load the feature scaler from file"""
        if scaler_path is None:
            scaler_path = self.scaler_path
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info("Feature scaler loaded successfully")
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            raise

    def get_training_data_set(self) -> dict[str, pd.DataFrame]:
        train_data = self.prepare_features_for_symbols(
            self.load_historical_data(
                self.symbols,
                ModelConfig.TRAIN_START_DATE,
                ModelConfig.TRAIN_END_DATE
            )
        )
        val_data = self.prepare_features_for_symbols(
            self.load_historical_data(
                self.symbols,
                ModelConfig.VAL_START_DATE,
                ModelConfig.VAL_END_DATE
            )
        )

        X_train, y_train = self.prepare_data_for_input(train_data, training=True)
        X_train_scaled = self.transform(X_train)
        X_val, y_val = self.prepare_data_for_input(val_data, training=True)
        X_val_scaled = self.transform(X_val)

        data_set = {
            'train_data': train_data,
            'val_data': val_data,
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_val': X_val_scaled,
            'y_val': y_val
        }
        
        return data_set
    
    def get_testing_data_set(self) -> dict[str, pd.DataFrame]:
        test_data = self.prepare_features_for_symbols(
            self.load_historical_data(
                self.symbols,
                ModelConfig.TEST_START_DATE,
                ModelConfig.TEST_END_DATE
            )
        )
        X_test, _ = self.prepare_data_for_input(test_data, training=False)
        data_set = {
            'test_data': test_data,
            'X_test': X_test
        }
        return data_set
    
    def get_transformed_input(self, data_input: Dict[str, pd.DataFrame]) -> np.ndarray:
        data = self.prepare_features_for_symbols(data_input)
        X, _ = self.prepare_data_for_input(data, training=False)
        return self.transform(X)

    def get_returns(self, data_X: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        return {symbol: data[ModelConfig.PRICE_COLUMN].shift(-1) / data[ModelConfig.PRICE_COLUMN] - 1
                for symbol, data in data_X.items()}

    def load_training_data(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        interval: str = '1h'
    ) -> Dict[str, pd.DataFrame]:
        data = {}
        symbols = TradingConfig.SYMBOLS
        for symbol in symbols:
            file_path = os.path.join(BaseConfig.DATA_PATH, interval, f"{symbol}.csv")
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['index'])
            df = df.set_index('datetime')
            df = df.loc[start_time:end_time]
            data[symbol] = df
        return data

    def save_training_data(self, data: Dict[str, pd.DataFrame]) -> None:
        for symbol, df in data.items():
            file_path = os.path.join(BaseConfig.DATA_PATH, BaseConfig.DATA_TIMEFRAME, f"{symbol}.csv")
            df.to_csv(file_path)