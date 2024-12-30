from typing import Dict, Optional
from portfolio import Portfolio
from lstm_model import CryptoLSTM
import numpy as np
import pandas as pd
from config import REBALANCE_THRESHOLD, SEQUENCE_LENGTH, REBALANCE_INTERVAL
import os
import tensorflow as tf
from strategy import Strategy

class ReturnMaximizationLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # Portfolio return (higher weight)
        returns = -1.5 * tf.reduce_mean(y_pred * y_true)
        
        # Sharpe ratio component
        returns_mean = tf.reduce_mean(y_pred * y_true)
        returns_std = tf.math.reduce_std(y_pred * y_true)
        sharpe = -returns_mean / (returns_std + 1e-7)
        
        # Transaction cost penalty
        turnover = tf.reduce_mean(tf.abs(y_pred[:, 1:] - y_pred[:, :-1]))
        transaction_cost = 0.1 * turnover
        
        return returns + 0.5 * sharpe + transaction_cost

class LSTMEnhancedStrategy(Strategy):
    def __init__(self, rebalance_threshold: float = REBALANCE_THRESHOLD, sequence_length: int = SEQUENCE_LENGTH):
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm_model = CryptoLSTM(sequence_length=sequence_length)
        self.rebalance_threshold = rebalance_threshold
        self.is_trained = False
        self.current_weights = None
        self.last_rebalance_time = None
        
    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk"""
        try:
            from lstm_model import CryptoLSTM, ReturnMaximizationLoss, StableMAE, WeightNormalization
            import tensorflow as tf
            
            # Create a new CryptoLSTM instance
            self.lstm_model = CryptoLSTM(sequence_length=self.sequence_length)
            
            # Load the Keras model with custom objects
            self.lstm_model.model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'ReturnMaximizationLoss': ReturnMaximizationLoss,
                    'StableMAE': StableMAE,
                    'WeightNormalization': WeightNormalization
                }
            )
            
            # Load the feature scaler if it was saved
            scaler_path = model_path.replace('.keras', '_scaler.pkl')
            if os.path.exists(scaler_path):
                import joblib
                self.lstm_model.feature_scaler = joblib.load(scaler_path)
            
            self.is_trained = True
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
    def save_model(self, model_path: str) -> None:
        """Save the current model and its scaler to disk"""
        try:
            if self.lstm_model is not None and self.lstm_model.model is not None:
                # Save the Keras model
                self.lstm_model.model.save(model_path)
                
                # Save the feature scaler
                import joblib
                scaler_path = model_path.replace('.keras', '_scaler.pkl')
                joblib.dump(self.lstm_model.feature_scaler, scaler_path)
                
                print(f"Successfully saved model to {model_path}")
                print(f"Successfully saved scaler to {scaler_path}")
            else:
                raise ValueError("No model to save. Train a model first.")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
            
    def train_model(self, historical_data: pd.DataFrame):
        """Train LSTM model with historical data"""
        train_metrics = self.lstm_model.train_model(historical_data)
        self.is_trained = True
        return train_metrics
    
    def evaluate(self, data: pd.DataFrame) -> Dict:
        """Evaluate model performance on given dataset"""
        if self.lstm_model is None:
            raise ValueError("Model must be trained before evaluation")

        # Prepare data
        X, y = self.lstm_model.prepare_data(data)
        
        # Get dimensions
        if len(X.shape) == 4:  # [samples, sequence_length, n_assets, n_features]
            n_samples, seq_len, n_assets, n_features = X.shape
            total_features = n_assets * n_features
            X = X.reshape(n_samples, seq_len, total_features)
        
        # Get the expected input shape from the model
        expected_features = self.lstm_model.model.input_shape[-1]
        expected_output = self.lstm_model.model.output_shape[-1]
        
        print(f"\nEvaluation debug:")
        print(f"Model's expected input features: {expected_features}")
        print(f"Model's expected output dimension: {expected_output}")
        print(f"Current input features: {X.shape[-1]}")
        print(f"Current output dimension: {y.shape[-1]}")
        print(f"Current shapes - X: {X.shape}, y: {y.shape}")
        
        # Handle input feature mismatch
        if X.shape[-1] != expected_features:
            if X.shape[-1] < expected_features:
                # Pad with zeros
                padding_size = expected_features - X.shape[-1]
                padding = np.zeros((X.shape[0], X.shape[1], padding_size))
                X = np.concatenate([X, padding], axis=-1)
                print(f"Padded X shape: {X.shape}")
            else:
                # Truncate
                X = X[:, :, :expected_features]
                print(f"Truncated X shape: {X.shape}")
        
        # Handle output dimension mismatch
        if y.shape[-1] != expected_output:
            if y.shape[-1] < expected_output:
                # Pad targets with zeros
                padding_size = expected_output - y.shape[-1]
                padding = np.zeros((y.shape[0], padding_size))
                y = np.concatenate([y, padding], axis=-1)
                print(f"Padded y shape: {y.shape}")
            else:
                # Truncate targets
                y = y[:, :expected_output]
                print(f"Truncated y shape: {y.shape}")
            
            # Renormalize weights after padding/truncating
            row_sums = y.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            y = y / row_sums
        
        # Ensure batch size is consistent
        batch_size = min(32, len(X))
        
        # Evaluate the model
        loss, mae = self.lstm_model.model.evaluate(X, y, batch_size=batch_size, verbose=0)
        
        return {
            'val_loss': loss,
            'val_mae': mae,
            'test_loss': loss,
            'test_mae': mae
        }
    
    def predict_weights(self, portfolio: Portfolio, current_prices: Dict[str, dict]) -> np.ndarray:
        """Predict optimal portfolio weights using the LSTM model"""
        if self.lstm_model is None:
            raise ValueError("Model not trained or loaded. Call train_model() or load_model() first.")
            
        return self.lstm_model.predict_weights(portfolio, current_prices)
    
    def get_weights(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                   timestamp: pd.Timestamp) -> Optional[Dict[str, float]]:
        """Get portfolio weights based on LSTM predictions"""
        try:
            # Check if we need to rebalance
            if (self.last_rebalance_time is None or 
                (timestamp - self.last_rebalance_time).total_seconds() >= REBALANCE_INTERVAL * 3600):
                
                # Get model predictions
                predicted_weights = self.predict_weights(portfolio, current_prices)
                if predicted_weights is None:
                    return None
                
                # Convert to dictionary
                self.current_weights = {
                    symbol: weight 
                    for symbol, weight in zip(portfolio.symbols, predicted_weights)
                }
                self.last_rebalance_time = timestamp
                
            return self.current_weights
            
        except Exception as e:
            print(f"Error in get_weights: {str(e)}")
            return None
    
    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                 timestep: int = 0) -> Dict[str, float]:
        """Enhanced minimum variance strategy using LSTM predictions"""
        # Calculate current portfolio value and weights
        current_equity, current_weights = self.calculate_current_weights(portfolio, current_prices)
        
        # Track portfolio state
        self.track_portfolio_state(portfolio, current_equity, timestep)
        
        if self.is_trained:
            predicted_weights = self.predict_weights(portfolio, current_prices)
            
            if predicted_weights is not None:
                max_deviation = max(abs(current_weights.get(symbol, 0) - weight) 
                                 for symbol, weight in zip(portfolio.symbols, predicted_weights))
                
                if max_deviation > self.rebalance_threshold:
                    weight_dict = dict(zip(portfolio.symbols[:len(predicted_weights)], predicted_weights))
                    signals = self.calculate_positions(portfolio, current_prices, weight_dict, current_equity)
                    return signals
        
        # If LSTM prediction fails or no rebalance needed, use optimization
        result = portfolio.get_optim_weights()
        if result is not None:
            optimal_weights = result.x
            weight_dict = dict(zip(portfolio.symbols[:len(optimal_weights)], optimal_weights))
            signals = self.calculate_positions(portfolio, current_prices, weight_dict, current_equity)
        else:
            # If optimization fails, maintain current positions
            signals = {}
            for symbol in portfolio.symbols:
                position = portfolio.portfolio_df.loc[symbol, 'position']
                if isinstance(position, pd.Series):
                    position = position.iloc[0]
                signals[symbol] = float(position)
        
        return signals