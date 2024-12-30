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
        print("\nTraining LSTM Enhanced Strategy...")
        
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
            print(f"Error during training: {str(e)}")
            raise
    
    def prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for LSTM training"""
        # Extract features
        features = pd.DataFrame(index=data.index)
        
        # Price and returns
        features['return'] = data['price'].pct_change()
        features['price'] = data['price']
        
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
        
        # Fill NaN values
        features = features.ffill().bfill()
        
        return features
    
    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                 timestep: int = 0) -> Dict[str, float]:
        """Execute LSTM enhanced strategy"""
        print("\n=== LSTM Enhanced Strategy Execution ===")
        
        try:
            # Calculate current portfolio value and weights
            current_equity, current_weights = self.calculate_current_weights(portfolio, current_prices)
            
            # Get price history and prepare features
            features = self.prepare_features(portfolio, current_prices)
            
            # Generate predictions
            predictions = self.model.predict(features)
            
            # Convert predictions to target weights
            target_weights = self.predictions_to_weights(predictions)
            
            # Calculate target positions
            signals = self.calculate_positions(portfolio, current_prices, target_weights, current_equity)
            
            # Track portfolio state
            self.track_portfolio_state(portfolio, current_equity, timestep)
            
            print("=== End Strategy Execution ===\n")
            return signals
            
        except Exception as e:
            print(f"Error executing strategy: {str(e)}")
            # Return current positions as fallback
            return {symbol: float(portfolio.portfolio_df.loc[symbol, 'position'])
                    for symbol in portfolio.portfolio_df.index}
    
    def prepare_features(self, portfolio: Portfolio, current_prices: Dict[str, dict]) -> pd.DataFrame:
        """Prepare features for prediction"""
        features = pd.DataFrame()
        
        for symbol, row in portfolio.portfolio_df.iterrows():
            try:
                # Get price history
                prices = pd.Series(row['asset'].get_price_history())
                if len(prices) < FeatureConfig.LOOKBACK_PERIOD:
                    continue
                
                # Calculate features
                symbol_features = pd.DataFrame(index=prices.index)
                
                # Price and returns
                symbol_features['return'] = prices.pct_change()
                symbol_features['price'] = prices
                
                # Technical indicators
                # Volatility
                symbol_features['volatility'] = symbol_features['return'].rolling(
                    window=FeatureConfig.LOOKBACK_PERIOD).std()
                
                # Momentum
                symbol_features['momentum'] = symbol_features['return'].rolling(
                    window=FeatureConfig.LOOKBACK_PERIOD).mean()
                
                # Price relative to SMA
                sma = symbol_features['price'].rolling(
                    window=FeatureConfig.LOOKBACK_PERIOD).mean()
                symbol_features['price_to_sma'] = symbol_features['price'] / sma
                symbol_features['price_std'] = symbol_features['price'].rolling(
                    window=FeatureConfig.LOOKBACK_PERIOD).std()
                
                # RSI
                delta = symbol_features['return']
                gain = (delta.where(delta > 0, 0)).rolling(
                    window=FeatureConfig.TECHNICAL_INDICATORS['RSI_PERIOD']).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(
                    window=FeatureConfig.TECHNICAL_INDICATORS['RSI_PERIOD']).mean()
                rs = gain / loss
                symbol_features['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD
                exp1 = symbol_features['price'].ewm(
                    span=FeatureConfig.TECHNICAL_INDICATORS['MACD_FAST'], adjust=False).mean()
                exp2 = symbol_features['price'].ewm(
                    span=FeatureConfig.TECHNICAL_INDICATORS['MACD_SLOW'], adjust=False).mean()
                symbol_features['macd'] = exp1 - exp2
                symbol_features['macd_signal'] = symbol_features['macd'].ewm(
                    span=FeatureConfig.TECHNICAL_INDICATORS['MACD_SIGNAL'], adjust=False).mean()
                
                # Bollinger Bands
                bb_sma = symbol_features['price'].rolling(
                    window=FeatureConfig.TECHNICAL_INDICATORS['BB_PERIOD']).mean()
                bb_std = symbol_features['price'].rolling(
                    window=FeatureConfig.TECHNICAL_INDICATORS['BB_PERIOD']).std()
                bb_upper = bb_sma + (FeatureConfig.TECHNICAL_INDICATORS['BB_STD'] * bb_std)
                bb_lower = bb_sma - (FeatureConfig.TECHNICAL_INDICATORS['BB_STD'] * bb_std)
                symbol_features['bb_position'] = (symbol_features['price'] - bb_lower) / (bb_upper - bb_lower)
                
                # Statistical moments
                symbol_features['skewness'] = symbol_features['return'].rolling(
                    window=FeatureConfig.LOOKBACK_PERIOD).skew()
                symbol_features['kurtosis'] = symbol_features['return'].rolling(
                    window=FeatureConfig.LOOKBACK_PERIOD).kurt()
                
                # Add symbol identifier
                symbol_features['symbol'] = symbol
                
                # Append to main features DataFrame
                features = pd.concat([features, symbol_features], ignore_index=True)
                
            except Exception as e:
                print(f"Error preparing features for {symbol}: {str(e)}")
                continue
        
        # Fill NaN values
        features = features.ffill().bfill()
        
        return features
    
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