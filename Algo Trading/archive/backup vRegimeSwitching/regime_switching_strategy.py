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
            self.regime_detector.fit(btc_data, btc_val_data)

            # Train LSTM model on all assets
            self.lstm_strategy.train_model(historical_data)
            
            # Print regime detection performance if validation data was used
            if btc_val_data is not None and not btc_val_data.empty:
                self._evaluate_regime_detection(btc_val_data)
            
        except Exception:
            raise
    
    def detect_regime(self, portfolio: Portfolio, current_prices: Dict[str, dict]) -> int:
        """Detect current market regime using BTC as the market indicator"""
        try:
            # Get BTC price history
            btc_asset = portfolio.portfolio_df.loc['BTCUSDT', 'asset']
            price_history = btc_asset.get_price_history()
            
            # Convert to Series and sort by timestamp
            timestamps = sorted(price_history.keys())
            prices = [float(price_history[t]) for t in timestamps]
            btc_prices = pd.Series(prices, index=timestamps)
            
            print(f"Original price series length: {len(btc_prices)}")
            
            # Extract features - this should account for the lookback period
            features = self.regime_detector.extract_features(btc_prices)
            print(f"Features array shape: {features.shape}")
            
            # Get the timestamp alignment after feature extraction
            max_lookback = RegimeConfig.get_max_lookback()
            feature_timestamps = btc_prices.index[max_lookback:]
            print(f"Feature timestamps length: {len(feature_timestamps)}")
            
            # Predict regimes
            regimes = self.regime_detector.hmm.predict(features)
            print(f"Regimes shape: {regimes.shape}")
            
            # Map regimes using return-based mapping
            raw_regimes = [self.regime_detector.regime_map[r] for r in regimes]
            
            # Apply regime smoothing
            smoothed_regimes = self._smooth_regime_transitions(raw_regimes)
            
            # Verify alignment
            if len(feature_timestamps) != len(smoothed_regimes):
                raise ValueError(
                    f"Timestamp misalignment after smoothing:\n"
                    f"Features: {len(feature_timestamps)}\n"
                    f"Smoothed regimes: {len(smoothed_regimes)}"
                )
            
            # Store history with proper alignment
            self.timestamps = list(feature_timestamps)
            self.regime_history = smoothed_regimes
            self.regime_probs_history = list(probs)
            
            # Use the most recent complete regime
            self.current_regime = smoothed_regimes[-1]
            
            return self.current_regime
            
        except Exception as e:
            raise RuntimeError(f"Error in detect_regime: {e}")
    
    def _evaluate_regime_detection(self, validation_data: pd.Series):
        """Evaluate regime detection performance on validation data"""
        try:
            # Get regime predictions on validation data
            features = self.regime_detector.extract_features(validation_data)
            regimes = self.regime_detector.hmm.predict(features)
            
            # Map regimes using Sharpe-based mapping
            regimes = np.array([self.regime_detector.regime_map[r] for r in regimes])
            
            # Calculate regime distribution
            regime_counts = np.bincount(regimes, minlength=self.n_regimes)
            regime_props = np.where(len(regimes) > 0,
                                  regime_counts / len(regimes),
                                  np.zeros_like(regime_counts))
            
            # Calculate transition matrix
            transitions = np.zeros((self.regime_detector.n_regimes, self.regime_detector.n_regimes))
            for i in range(len(regimes)-1):
                transitions[regimes[i], regimes[i+1]] += 1
            
            # Normalize transition matrix with safety check
            row_sums = transitions.sum(axis=1, keepdims=True)
            transition_matrix = np.where(row_sums > 0,
                                       transitions / row_sums,
                                       np.zeros_like(transitions))
            
            # Calculate performance metrics by regime
            returns = validation_data.pct_change()
            
        except Exception:
            pass
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        # Add safety check for drawdown calculation
        drawdown = np.where(running_max > 0,
                           (cumulative - running_max) / running_max,
                           0)
        return abs(float(np.nanmin(drawdown)))
    
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
            
            # Normalize weights with safety check
            total_weight = sum(combined_weights.values())
            if total_weight > 0:
                combined_weights = {
                    symbol: weight / total_weight 
                    for symbol, weight in combined_weights.items()
                }
            else:
                # Fallback to equal weights if total weight is zero
                weight = 1.0 / len(self.symbols)
                combined_weights = {symbol: weight for symbol in self.symbols}
            
            return combined_weights
            
        except Exception:
            # Return equal weights as fallback
            weight = 1.0 / len(self.symbols)
            return {symbol: weight for symbol in self.symbols}
    
    def __call__(self, portfolio: Portfolio, current_prices: Dict[str, dict], 
                 timestep: int = 0) -> Dict[str, float]:
        """Execute regime switching strategy"""
        try:
            # Calculate current portfolio value and weights
            current_equity, current_weights = self.calculate_current_weights(portfolio, current_prices)

            # Detect current regime
            regime = self.detect_regime(portfolio, current_prices)

            # Get weights from sub-strategies
            lstm_weights = self.lstm_strategy(portfolio, current_prices, timestep)
            equal_weights = self.equal_weight_strategy(portfolio, current_prices, timestep)
            
            # Combine weights based on regime
            target_weights = self.get_regime_weights(regime, lstm_weights, equal_weights)

            # Calculate target positions
            signals = self.calculate_positions(portfolio, current_prices, target_weights, current_equity)

            # Track portfolio state and timestamp
            self.track_portfolio_state(portfolio, current_equity, timestep)

            return signals
            
        except Exception as e:
            print(f"Error in __call__: {e}")
            # Return current positions as fallback
            signals = {}
            for symbol in self.symbols:
                position = portfolio.portfolio_df.loc[symbol, 'size']
                if isinstance(position, pd.Series):
                    position = position.iloc[0]
                signals[symbol] = float(position)
            return signals
    
    def get_regime_stats(self) -> pd.DataFrame:
        """Get statistics about regime transitions and probabilities"""
        try:
            if not self.regime_history or not self.regime_probs_history or not self.timestamps:
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
            n_regimes = len(self.regime_history)
            n_probs = len(self.regime_probs_history)
            n_timestamps = len(self.timestamps)
            print(f"Error in get_regime_stats: {e}")
            print(f"Regime history length: {n_regimes}")
            print(f"Regime probs history length: {n_probs}")
            print(f"Timestamps length: {n_timestamps}")
            quit()
            return pd.DataFrame() 
    
    def _smooth_regime_transitions(self, raw_regimes: List[int], window: Optional[int] = None) -> List[int]:
        """
        Smooth regime transitions to prevent frequent switching
        
        Args:
            raw_regimes: List of raw regime predictions
            window: Optional override for window size
        
        Returns:
            List of smoothed regime predictions
        """
        try:
            # Get smoothing parameters from config
            smoothing_params = RegimeConfig.get_smoothing_params()
            window = window or smoothing_params['WINDOW']
            threshold = smoothing_params['THRESHOLD']
            
            smoothed_regimes = []
            for i in range(len(raw_regimes)):
                # Get the window of regimes up to current point
                start_idx = max(0, i - window + 1)
                regime_window = raw_regimes[start_idx:i + 1]
                
                if len(regime_window) < window // 2:
                    # Not enough history, use raw regime
                    smoothed_regimes.append(raw_regimes[i])
                    continue
                
                # Count regime occurrences in window
                regime_counts = {}
                for regime in regime_window:
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                # Calculate regime probabilities
                total_count = len(regime_window)
                regime_probs = {
                    regime: count / total_count 
                    for regime, count in regime_counts.items()
                }
                
                # Only switch regime if new regime is dominant
                current_regime = smoothed_regimes[-1] if smoothed_regimes else raw_regimes[i]
                
                # Find most probable regime
                most_probable = max(regime_probs.items(), key=lambda x: x[1])
                
                if most_probable[1] >= threshold:
                    # Strong enough evidence to switch
                    smoothed_regimes.append(most_probable[0])
                else:
                    # Maintain current regime
                    smoothed_regimes.append(current_regime)
            
            return smoothed_regimes
            
        except Exception as e:
            print(f"Error in regime smoothing: {e}")
            return raw_regimes  # Return unsmoothed regimes as fallback 