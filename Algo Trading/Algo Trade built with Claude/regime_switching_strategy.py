from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from strategy import Strategy
from lstm_enhanced_strategy import LSTMEnhancedStrategy
from equal_weight_strategy import EqualWeightStrategy
from regime_detector_bull_bear import RegimeDetectorBullBear
from portfolio import Portfolio
from config import RegimeConfig

class RegimeSwitchingStrategy(Strategy):
    """Strategy that switches between different strategies based on market regime"""
    
    def __init__(self, symbols: List[str], lookback_period: int = RegimeConfig.LOOKBACK_PERIOD):
        super().__init__()
        self.symbols = symbols
        self.regime_detector = RegimeDetectorBullBear(
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
        self.timestamps = []
        
        # Ensure BTC is in the symbol list
        if 'BTCUSDT' not in symbols:
            raise ValueError("BTCUSDT must be included in the symbol list for regime detection")
    
    def train(self, historical_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None):
        """Train both the regime detector and LSTM model"""
        print("\nTraining Regime Switching Strategy...")
        
        try:
            # Extract BTC data for regime detection
            # Data is indexed by datetime with multiple columns per symbol
            btc_data = pd.Series(index=historical_data.index)
            btc_val_data = pd.Series(index=validation_data.index) if validation_data is not None else None
            
            # Find BTC price column
            btc_cols = [col for col in historical_data.columns if 'BTCUSDT' in col and 'price' in col]
            if not btc_cols:
                raise ValueError("No BTC price data found in historical data")
            
            btc_col = btc_cols[0]
            btc_data = historical_data[btc_col]
            
            if validation_data is not None:
                btc_val_data = validation_data[btc_col]
            
            if btc_data.empty:
                raise ValueError("No BTC price data found in historical data")
            
            # Train regime detector using BTC price data
            print("Training Regime Detector on BTC data...")
            self.regime_detector.fit(btc_data, btc_val_data)
            
            # Train LSTM model on all assets
            print("Training LSTM Model...")
            self.lstm_strategy.train_model(historical_data)
            
            print("Training complete!")
            
            # Print regime detection performance if validation data was used
            if btc_val_data is not None and not btc_val_data.empty:
                self._evaluate_regime_detection(btc_val_data)
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def detect_regime(self, portfolio: Portfolio, current_prices: Dict[str, dict]) -> int:
        """Detect current market regime using BTC as the market indicator"""
        try:
            # Get BTC price history
            btc_asset = portfolio.portfolio_df.loc['BTCUSDT', 'asset']
            if btc_asset is None:
                print("Warning: BTC price history not available")
                return 1  # Default to neutral regime
            
            btc_prices = pd.Series(btc_asset.get_price_history())
            if btc_prices.empty:
                print("Warning: Empty BTC price history")
                return 1
            
            # Extract features and get regime predictions for all time steps
            features = self.regime_detector.extract_features(btc_prices)
            regimes = self.regime_detector.hmm.predict(features)
            probs = self.regime_detector.hmm.predict_proba(features)
            
            # Map regimes using return-based mapping
            regimes = [self.regime_detector.regime_map[r] for r in regimes]
            
            # Store regime information
            self.current_regime = regimes[-1]
            self.regime_history = regimes
            self.regime_probs_history = list(probs)
            
            # Store timestamps
            self.timestamps = btc_prices.index
            
            print(f"\nCurrent Market Regime: {self.regime_detector.regime_labels[self.current_regime]}")
            print(f"Regime Probabilities: Bull: {probs[-1][0]:.2%}, Neutral: {probs[-1][1]:.2%}, Bear: {probs[-1][2]:.2%}")
            
            return self.current_regime
            
        except Exception as e:
            print(f"Error detecting regime: {str(e)}")
            return 1  # Default to neutral regime
    
    def _evaluate_regime_detection(self, validation_data: pd.Series):
        """Evaluate regime detection performance on validation data"""
        try:
            # Get regime predictions on validation data
            features = self.regime_detector.extract_features(validation_data)
            regimes = self.regime_detector.hmm.predict(features)
            
            # Map regimes using Sharpe-based mapping
            regimes = np.array([self.regime_detector.regime_map[r] for r in regimes])
            
            # Calculate regime distribution
            regime_counts = np.bincount(regimes, minlength=self.regime_detector.n_regimes)
            regime_props = regime_counts / len(regimes)
            
            print("\nRegime Detection Performance on BTC Validation Data:")
            for i, (count, prop) in enumerate(zip(regime_counts, regime_props)):
                print(f"{self.regime_detector.regime_labels[i]}:")
                print(f"  Count: {count}")
                print(f"  Proportion: {prop:.2%}")
            
            # Calculate transition matrix
            transitions = np.zeros((self.regime_detector.n_regimes, self.regime_detector.n_regimes))
            for i in range(len(regimes)-1):
                transitions[regimes[i], regimes[i+1]] += 1
            
            # Normalize transition matrix
            row_sums = transitions.sum(axis=1, keepdims=True)
            transition_matrix = transitions / row_sums
            
            print("\nRegime Transition Probabilities:")
            for i in range(self.regime_detector.n_regimes):
                for j in range(self.regime_detector.n_regimes):
                    if transition_matrix[i, j] > 0:
                        print(f"{self.regime_detector.regime_labels[i]} -> "
                              f"{self.regime_detector.regime_labels[j]}: "
                              f"{transition_matrix[i, j]:.2%}")
            
            # Calculate performance metrics by regime
            returns = validation_data.pct_change()
            print("\nRegime Performance Metrics:")
            for i in range(self.regime_detector.n_regimes):
                regime_mask = (regimes == i)
                regime_rets = returns[regime_mask]
                if len(regime_rets) > 0:
                    print(f"\n{self.regime_detector.regime_labels[i]}:")
                    print(f"  Mean Return: {np.mean(regime_rets):.2%}")
                    print(f"  Volatility: {np.std(regime_rets):.2%}")
                    print(f"  Sharpe Ratio: {np.mean(regime_rets)/np.std(regime_rets) if np.std(regime_rets) > 0 else 0:.2f}")
                    print(f"  Max Drawdown: {self._calculate_max_drawdown(regime_rets):.2%}")
            
        except Exception as e:
            print(f"Error evaluating regime detection: {str(e)}")
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(float(drawdown.min()))
    
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
            
            # Track portfolio state and timestamp
            self.track_portfolio_state(portfolio, current_equity, timestep)
            
            # Store timestamp
            if current_prices:
                first_symbol = list(current_prices.keys())[0]
                timestamp = pd.to_datetime(current_prices[first_symbol]['time'], unit='ms')
                self.timestamps.append(timestamp)
            
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
        """Get statistics about regime transitions and probabilities"""
        try:
            if not self.regime_history or not self.regime_probs_history or not self.timestamps:
                print("Warning: No regime history available")
                return pd.DataFrame()
                
            # Convert history to DataFrame
            df = pd.DataFrame({
                'regime': [self.regime_detector.regime_labels[r] for r in self.regime_history],
                'bull_prob': [probs[0] for probs in self.regime_probs_history],
                'neutral_prob': [probs[1] for probs in self.regime_probs_history],
                'bear_prob': [probs[2] for probs in self.regime_probs_history]
            }, index=self.timestamps)
            
            return df
            
        except Exception as e:
            print(f"Error calculating regime stats: {str(e)}")
            return pd.DataFrame() 