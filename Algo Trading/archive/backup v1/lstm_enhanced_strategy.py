from typing import Dict, Optional
from portfolio import Portfolio
from lstm_model import CryptoLSTM
import numpy as np
import pandas as pd
from config import REBALANCE_THRESHOLD, SEQUENCE_LENGTH, REBALANCE_INTERVAL
import os
import tensorflow as tf

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

class LSTMEnhancedStrategy:
    def __init__(self, rebalance_threshold: float = REBALANCE_THRESHOLD, sequence_length: int = SEQUENCE_LENGTH):
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
        signals = {}
        current_equity = portfolio.cash
        current_weights = {}
        
        # Track portfolio and cash values
        if not hasattr(self, 'portfolio_values'):
            self.portfolio_values = []
            self.cash_values = []
            self.timestamps = []
        
        # Calculate current equity and weights
        for symbol in portfolio.symbols:
            position = portfolio.portfolio_df.loc[symbol, 'position']
            if isinstance(position, pd.Series):
                position = position.iloc[0]
            position_value = float(position) * float(current_prices[symbol]['markPrice'])
            current_equity += position_value
            current_weights[symbol] = position_value / current_equity if current_equity > 0 else 0
        
        # Store current values
        self.portfolio_values.append(current_equity)
        self.cash_values.append(portfolio.cash)
        self.timestamps.append(pd.Timestamp.now() if timestep == 0 else pd.Timestamp.now() + pd.Timedelta(hours=timestep))
        
        # Function to calculate positions while respecting cash constraints
        def calculate_positions(weights_dict):
            available_cash = portfolio.cash
            positions = {}
            total_target_value = current_equity  # Total portfolio target value
            
            # First pass: calculate all desired positions and cash requirements
            total_buy_value = 0
            total_sell_value = 0
            
            for symbol in portfolio.symbols:
                if symbol in weights_dict:
                    price = float(current_prices[symbol]['markPrice'])
                    
                    # Handle position access properly
                    position_value = portfolio.portfolio_df.loc[symbol, 'position']
                    if isinstance(position_value, pd.Series):
                        current_pos = float(position_value.iloc[0])
                    else:
                        current_pos = float(position_value)
                    
                    # Calculate target value for this asset
                    target_value = float(weights_dict[symbol] * total_target_value)
                    current_value = current_pos * price
                    value_diff = target_value - current_value
                    
                    positions[symbol] = {
                        'price': price,
                        'current_pos': current_pos,
                        'target_value': target_value,
                        'value_diff': value_diff
                    }
                    
                    if value_diff > 0:
                        total_buy_value += value_diff
                    else:
                        total_sell_value += abs(value_diff)
            
            # Calculate scaling factor if we need more cash than available
            available_cash_with_sells = available_cash + total_sell_value
            if total_buy_value > available_cash_with_sells:
                buy_scale_factor = available_cash_with_sells / total_buy_value
            else:
                buy_scale_factor = 1.0
                
            # Second pass: calculate final positions with scaling
            final_positions = {}
            total_final_value = 0
            
            for symbol, pos_info in positions.items():
                price = pos_info['price']
                current_pos = pos_info['current_pos']
                value_diff = pos_info['value_diff']
                
                if value_diff > 0:
                    # Scale down buys if necessary
                    scaled_value_diff = value_diff * buy_scale_factor
                else:
                    # Keep sells as is
                    scaled_value_diff = value_diff
                
                # Calculate final position
                target_value = current_pos * price + scaled_value_diff
                final_position = target_value / price
                final_positions[symbol] = final_position
                
                total_final_value += final_position * price
                
            return final_positions
        
        if self.is_trained:
            predicted_weights = self.predict_weights(portfolio, current_prices)
            
            if predicted_weights is not None:
                max_deviation = max(abs(current_weights.get(symbol, 0) - weight) 
                                 for symbol, weight in zip(portfolio.symbols, predicted_weights))
                
                if max_deviation > self.rebalance_threshold:
                    weight_dict = dict(zip(portfolio.symbols[:len(predicted_weights)], predicted_weights))
                    signals = calculate_positions(weight_dict)
                    return signals
        
        result = portfolio.get_optim_weights()
        if result is not None:
            optimal_weights = result.x
            weight_dict = dict(zip(portfolio.symbols[:len(optimal_weights)], optimal_weights))
            signals = calculate_positions(weight_dict)
        else:
            # If optimization fails, maintain current positions
            for symbol in portfolio.symbols:
                position = portfolio.portfolio_df.loc[symbol, 'position']
                if isinstance(position, pd.Series):
                    position = position.iloc[0]
                signals[symbol] = float(position)
        
        # Plot performance if we have enough data points
        if len(self.portfolio_values) > 1:
            portfolio_series = pd.Series(self.portfolio_values, index=self.timestamps)
            cash_series = pd.Series(self.cash_values, index=self.timestamps)
            self.plot_portfolio_performance(portfolio_series, cash_series)
        
        return signals
    
    def plot_portfolio_performance(self, portfolio_values: pd.Series, cash_values: pd.Series):
        """Plot portfolio value and cash position over time"""
        import matplotlib.pyplot as plt
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot total portfolio value
        ax1.plot(portfolio_values.index, portfolio_values.values, 'b-', label='Portfolio Value')
        ax1.set_title('Total Portfolio Value Over Time')
        ax1.set_ylabel('Value (USD)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot cash position
        ax2.plot(cash_values.index, cash_values.values, 'g-', label='Cash Position')
        ax2.set_title('Cash Position Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value (USD)')
        ax2.grid(True)
        ax2.legend()
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        fig.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
        plt.close()