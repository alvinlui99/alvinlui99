from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from strategy import Strategy
from lstm_enhanced_strategy import LSTMEnhancedStrategy
from equal_weight_strategy import EqualWeightStrategy
from regime_detector import RegimeDetector
from portfolio import Portfolio
from config import RegimeConfig

class RegimeSwitchingStrategy(Strategy):
    """Strategy that switches between different strategies based on market regime"""
    
    def __init__(self, symbols: List[str], lookback_period: int = RegimeConfig.LOOKBACK_PERIOD):
        super().__init__()
        self.symbols = symbols
        self.regime_detector = RegimeDetector(
            n_regimes=RegimeConfig.N_REGIMES,
            lookback_period=lookback_period
        )
        
        # Initialize sub-strategies
        self.lstm_strategy = LSTMEnhancedStrategy()
        self.equal_weight_strategy = EqualWeightStrategy()
        
        # Track current regime and history
        self.current_regime = None
        self.regime_history = []
        self.regime_probs_history = []
        
    def train(self, historical_data: pd.DataFrame):
        """Train both the regime detector and LSTM model"""
        print("\nTraining Regime Switching Strategy...")
        
        try:
            # Ensure we have price data
            if 'price' not in historical_data.columns:
                raise ValueError("Historical data must contain 'price' column")
            
            # Train regime detector using price data
            print("Training Regime Detector...")
            price_data = historical_data['price'].copy()
            self.regime_detector.fit(price_data)
            
            # Train LSTM model
            print("Training LSTM Model...")
            self.lstm_strategy.train_model(historical_data)
            
            print("Training complete!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        
    def detect_regime(self, portfolio: Portfolio, current_prices: Dict[str, dict]) -> int:
        """Detect current market regime"""
        try:
            # Get price history from portfolio
            price_history = []
            for symbol in self.symbols:
                asset = portfolio.portfolio_df.loc[symbol, 'asset']
                if asset is None:
                    continue
                    
                prices = pd.Series(asset.get_price_history())
                if not prices.empty:
                    price_history.append(prices)
            
            if not price_history:
                print("Warning: No price history available, using default regime")
                return 1  # Default to medium volatility regime
                
            # Use average price across assets
            avg_prices = pd.concat(price_history, axis=1).mean(axis=1)
            
            # Detect regime
            regime, probs = self.regime_detector.predict_regime(avg_prices)
            
            # Store regime information
            self.current_regime = regime
            self.regime_history.append(regime)
            self.regime_probs_history.append(probs)
            
            return regime
            
        except Exception as e:
            print(f"Error detecting regime: {str(e)}")
            return 1  # Default to medium volatility regime
    
    def get_regime_weights(self, regime: int, lstm_weights: Dict[str, float], 
                          equal_weights: Dict[str, float]) -> Dict[str, float]:
        """Get optimal weights based on current regime"""
        try:
            # Get base weights for regime
            base_weights = self.regime_detector.get_optimal_weights(regime, self.symbols)
            
            # Get mixing weights for current regime
            mix = RegimeConfig.REGIME_MIXING[regime]
            
            # Combine weights
            combined_weights = {}
            for symbol in self.symbols:
                lstm_weight = lstm_weights.get(symbol, 0)
                equal_weight = equal_weights.get(symbol, 0)
                base_weight = base_weights.get(symbol, 0)
                
                # Ensure weights are numeric
                lstm_weight = float(lstm_weight)
                equal_weight = float(equal_weight)
                base_weight = float(base_weight)
                
                # Weighted average of strategy weights
                strategy_weight = (mix['lstm'] * lstm_weight + 
                                 mix['equal'] * equal_weight)
                
                # Combine with regime base weights
                combined_weights[symbol] = (RegimeConfig.STRATEGY_WEIGHT_RATIO * strategy_weight + 
                                         RegimeConfig.BASE_WEIGHT_RATIO * base_weight)
            
            # Normalize weights
            total_weight = sum(combined_weights.values())
            if total_weight > 0:
                combined_weights = {
                    symbol: weight / total_weight 
                    for symbol, weight in combined_weights.items()
                }
            
            return combined_weights
            
        except Exception as e:
            print(f"Error calculating regime weights: {str(e)}")
            # Return equal weights as fallback
            weight = 1.0 / len(self.symbols)
            return {symbol: weight for symbol in self.symbols}
    
    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                 timestep: int = 0) -> Dict[str, float]:
        """Execute regime switching strategy"""
        print("\n=== Regime Switching Strategy Execution ===")
        
        try:
            # Calculate current portfolio value and weights
            current_equity, current_weights = self.calculate_current_weights(portfolio, current_prices)
            
            # Detect current regime
            regime = self.detect_regime(portfolio, current_prices)
            print(f"Current Regime: {self.regime_detector.regime_labels[regime]}")
            
            # Get weights from sub-strategies
            lstm_weights = self.lstm_strategy(portfolio, current_prices, timestep)
            equal_weights = self.equal_weight_strategy(portfolio, current_prices, timestep)
            
            # Combine weights based on regime
            target_weights = self.get_regime_weights(regime, lstm_weights, equal_weights)
            
            # Calculate target positions
            signals = self.calculate_positions(portfolio, current_prices, target_weights, current_equity)
            
            # Track portfolio state
            self.track_portfolio_state(portfolio, current_equity, timestep)
            
            print("=== End Strategy Execution ===\n")
            return signals
            
        except Exception as e:
            print(f"Error executing strategy: {str(e)}")
            # Return current positions as fallback
            signals = {}
            for symbol in self.symbols:
                position = portfolio.portfolio_df.loc[symbol, 'position']
                if isinstance(position, pd.Series):
                    position = position.iloc[0]
                signals[symbol] = float(position)
            return signals
    
    def get_regime_stats(self) -> pd.DataFrame:
        """Get statistics about regime transitions and durations"""
        try:
            if not self.regime_history:
                return pd.DataFrame(columns=['regime', 'count', 'avg_duration', 'max_duration', 'pct_time'])
                
            # Convert history to DataFrame
            df = pd.DataFrame({
                'regime': [self.regime_detector.regime_labels[r] for r in self.regime_history],
                'timestamp': self.timestamps
            })
            
            # Calculate regime durations
            df['regime_change'] = df['regime'].ne(df['regime'].shift())
            df['regime_duration'] = df.groupby(df['regime_change'].cumsum()).cumcount() + 1
            
            # Calculate statistics
            stats = []
            for regime in range(self.regime_detector.n_regimes):
                regime_label = self.regime_detector.regime_labels[regime]
                regime_data = df[df['regime'] == regime_label]
                stats.append({
                    'regime': regime_label,
                    'count': len(regime_data),
                    'avg_duration': float(regime_data['regime_duration'].mean()),
                    'max_duration': int(regime_data['regime_duration'].max()),
                    'pct_time': float(len(regime_data) / len(df) * 100)
                })
            
            return pd.DataFrame(stats)
            
        except Exception as e:
            print(f"Error calculating regime stats: {str(e)}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['regime', 'count', 'avg_duration', 'max_duration', 'pct_time']) 