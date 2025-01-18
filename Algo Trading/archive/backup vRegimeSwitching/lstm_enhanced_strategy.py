from typing import Dict, Optional
import numpy as np
import pandas as pd
from strategy import Strategy
from portfolio import Portfolio
from lstm_model import LSTMModel
from config import ModelConfig, FeatureConfig, MODEL_PATH

class LSTMEnhancedStrategy(Strategy):
    """Strategy that uses LSTM predictions to determine portfolio weights"""
    
    def __init__(self):
        super().__init__()
        self.model = LSTMModel()
        self.feature_columns = None
        self.target_columns = None
        self.scaler = None
        
    def train_model(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Train LSTM model on historical data"""
        try:
            # Prepare data for training
            train_data = self.prepare_training_data(historical_data)
            
            # Split data into train and validation sets
            split_idx = int(len(train_data) * ModelConfig.TRAIN_SIZE)
            train_set = train_data[:split_idx]
            val_set = train_data[split_idx:]
            
            # Train model
            metrics = self.model.train(train_set, val_set)
            
            # Save model if training was successful
            if metrics['val_loss'] < float('inf'):
                self.model.save_model(MODEL_PATH)
            
            return metrics
            
        except Exception as e:
            raise
    
    def _calculate_features(self, prices: pd.Series, symbol: str) -> pd.DataFrame:
        """Calculate features for a single symbol's price series"""
        try:
            features = pd.DataFrame(index=prices.index)
            
            # Price and returns
            features['return'] = prices.pct_change()
            features['price'] = prices
            
            # Technical indicators
            # Volatility
            features['volatility'] = features['return'].rolling(
                window=FeatureConfig.LOOKBACK_PERIOD).std()
            
            # Momentum
            features['momentum'] = features['return'].rolling(
                window=FeatureConfig.LOOKBACK_PERIOD).mean()
            
            # Price relative to SMA
            sma = features['price'].rolling(window=FeatureConfig.LOOKBACK_PERIOD).mean()
            features['price_to_sma'] = features['price'] / sma
            features['price_std'] = features['price'].rolling(
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
            exp1 = features['price'].ewm(
                span=FeatureConfig.TECHNICAL_INDICATORS['MACD_FAST'], adjust=False).mean()
            exp2 = features['price'].ewm(
                span=FeatureConfig.TECHNICAL_INDICATORS['MACD_SLOW'], adjust=False).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(
                span=FeatureConfig.TECHNICAL_INDICATORS['MACD_SIGNAL'], adjust=False).mean()
            
            # Bollinger Bands
            bb_sma = features['price'].rolling(
                window=FeatureConfig.TECHNICAL_INDICATORS['BB_PERIOD']).mean()
            bb_std = features['price'].rolling(
                window=FeatureConfig.TECHNICAL_INDICATORS['BB_PERIOD']).std()
            bb_upper = bb_sma + (FeatureConfig.TECHNICAL_INDICATORS['BB_STD'] * bb_std)
            bb_lower = bb_sma - (FeatureConfig.TECHNICAL_INDICATORS['BB_STD'] * bb_std)
            features['bb_position'] = (features['price'] - bb_lower) / (bb_upper - bb_lower)
            
            # Statistical moments
            features['skewness'] = features['return'].rolling(
                window=FeatureConfig.LOOKBACK_PERIOD).skew()
            features['kurtosis'] = features['return'].rolling(
                window=FeatureConfig.LOOKBACK_PERIOD).kurt()
            
            # Add symbol identifier
            features['symbol'] = symbol
            
            # Prefix all columns (except 'symbol') with symbol name
            features.columns = [f'{symbol}_{col}' if col != 'symbol' else col 
                              for col in features.columns]
            
            return features
            
        except Exception as e:
            return None

    def prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for LSTM training from historical data DataFrame"""
        all_features = []
        
        # Get unique symbols by finding price columns
        symbols = set()
        for col in data.columns:
            if col.endswith('_price'):
                symbol = col[:-6]  # Remove '_price' to get symbol
                symbols.add(symbol)
        
        if not symbols:
            raise ValueError("No price data found in historical data")
        
        for symbol in symbols:
            price_col = f'{symbol}_price'
            if price_col not in data.columns:
                continue
            
            if len(data[price_col]) < FeatureConfig.LOOKBACK_PERIOD:
                continue
                
            features = self._calculate_features(data[price_col], symbol)
            if features is None or features.empty:
                raise ValueError(f"No valid features calculated for symbol")
            all_features.append(features)
        
        if not all_features:
            raise ValueError("No valid features could be calculated for any symbol")
        
        # Filter out empty DataFrames before concatenation
        valid_features = [df for df in all_features if not df.empty]
        if not valid_features:
            raise ValueError("No valid features available after filtering")
        
        # Combine features from all symbols
        combined_features = pd.concat(valid_features, axis=1)
        combined_features = combined_features.ffill().bfill()
        
        # Drop non-numerical columns and any columns with NaN values
        numerical_features = combined_features.select_dtypes(include=[np.number])
        numerical_features = numerical_features.dropna(axis=1, how='any')
        
        if numerical_features.empty:
            raise ValueError("No valid numerical features available after preprocessing")
        
        return numerical_features
    
    def prepare_features(self, portfolio: Portfolio, current_prices: Dict[str, dict]) -> pd.DataFrame:
        """Prepare features for prediction from portfolio data"""
        all_features = []
        
        for symbol, row in portfolio.portfolio_df.iterrows():
            try:
                prices = pd.Series(row['asset'].get_price_history())
                if len(prices) < FeatureConfig.LOOKBACK_PERIOD:
                    continue
                
                features = self._calculate_features(prices, symbol)
                if features is None or features.empty:
                    raise ValueError(f"No valid features calculated for {symbol}")
                all_features.append(features)
                    
            except Exception as e:
                continue
        
        # Filter out empty DataFrames before concatenation
        valid_features = [df for df in all_features if not df.empty]
        
        # Combine all features
        if valid_features:
            combined_features = pd.concat(valid_features, axis=1)
            combined_features = combined_features.ffill().bfill()
            
            # Drop non-numerical columns and any columns with NaN values
            numerical_features = combined_features.select_dtypes(include=[np.number])
            numerical_features = numerical_features.dropna(axis=1, how='any')
            
            if numerical_features.empty:
                return pd.DataFrame()
                
            return numerical_features
        else:
            return pd.DataFrame()
    
    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                 timestep: int = 0) -> Dict[str, float]:
        """Execute LSTM enhanced strategy"""
        try:
            # Calculate current portfolio value and weights
            current_equity, current_weights = self.calculate_current_weights(portfolio, current_prices)
            
            # Get price history and prepare features
            features = self.prepare_features(portfolio, current_prices)
            
            # Generate predictions
            predictions = self.model.predict(features, verbose=0)
            
            # Convert predictions to target weights
            target_weights = self.predictions_to_weights(predictions)
            
            # Calculate target positions
            signals = self.calculate_positions(portfolio, current_prices, target_weights, current_equity)
            
            # Track portfolio state
            self.track_portfolio_state(portfolio, current_equity, timestep)
            
            return signals
            
        except Exception as e:
            # Return current positions as fallback
            return {symbol: float(portfolio.portfolio_df.loc[symbol, 'size'])
                    for symbol in portfolio.portfolio_df.index}
    
    def predictions_to_weights(self, predictions: np.ndarray) -> Dict[str, float]:
        """Convert model predictions to portfolio weights"""
        # Ensure predictions are positive
        predictions = np.maximum(predictions, 0)
        
        # Normalize to sum to 1
        total_pred = np.sum(predictions)
        if total_pred > 0:
            weights = predictions / total_pred
        else:
            # If all predictions are 0, use equal weights
            weights = np.ones(len(predictions)) / len(predictions)
        
        # Convert to dictionary
        weight_dict = {
            symbol: float(weight)
            for symbol, weight in zip(self.portfolio.portfolio_df.index, weights)
        }
        
        return weight_dict