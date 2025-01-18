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
        
    def train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, float]:
        """Train LSTM model on historical data with separate validation set"""
        print("\nTraining LSTM Enhanced Strategy...")
        
        try:
            # Train model with validation data
            print("Training LSTM model...")
            metrics = self.model.train(train_data, val_data)
            
            # Save model if training was successful
            if metrics['val_loss'] < float('inf'):
                self.model.save_model(MODEL_PATH)
            
            return metrics
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, float], 
                 features_data: Dict[str, pd.DataFrame], timestep: int = 0) -> Dict[str, float]:
        """Execute LSTM enhanced strategy"""
        print("\n=== LSTM Enhanced Strategy Execution ===")
        
        try:
            # Calculate current portfolio value and weights
            current_equity, current_weights = self.calculate_current_weights(portfolio, current_prices)
            
            # Get current features from each symbol's DataFrame
            current_features = {
                symbol: df.iloc[timestep:timestep+1] 
                for symbol, df in features_data.items()
            }
            
            # Combine features for LSTM input
            combined_features = pd.concat(current_features.values(), axis=1)
            
            # Generate predictions
            predictions = self.model.predict(combined_features)
            
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
            return {symbol: float(portfolio.get_position(symbol))
                    for symbol in portfolio.portfolio_df.index}
    
    def prepare_features(self, portfolio: Portfolio, current_prices: Dict[str, dict]) -> pd.DataFrame:
        """Prepare features for prediction"""
        features = pd.DataFrame()
        
        for symbol, row in portfolio.portfolio_df.iterrows():
            try:
                # Get price history from asset object
                prices = pd.Series(row['asset'].get_price_history())
                if len(prices) < FeatureConfig.LOOKBACK_PERIOD:
                    continue
                
                # Create price DataFrame
                price_df = pd.DataFrame({'price': prices})
                
                # Create features using data_utils
                from data_utils import engineer_features
                symbol_features = engineer_features(price_df, 'price')
                
                # Rename columns to include symbol identifier
                symbol_features.columns = [f'{col}_{symbol}' for col in symbol_features.columns]
                
                # Add symbol column
                symbol_features['symbol'] = symbol
                
                # Append to main features DataFrame
                features = pd.concat([features, symbol_features], ignore_index=True)
                
            except Exception as e:
                raise Exception(f"Error preparing features for {symbol}: {str(e)}")
        
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