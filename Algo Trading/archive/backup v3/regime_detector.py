from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from config import RegimeConfig

class RegimeDetector:
    """Detect market regimes using Hidden Markov Models"""
    
    def __init__(self, n_regimes: int = RegimeConfig.N_REGIMES, 
                 lookback_period: int = RegimeConfig.LOOKBACK_PERIOD):
        """
        Initialize regime detector
        
        Args:
            n_regimes: Number of market regimes to detect
            lookback_period: Number of periods to use for regime detection
        """
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.hmm = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type=RegimeConfig.HMM_COVARIANCE_TYPE,
            n_iter=RegimeConfig.HMM_N_ITER
        )
        self.scaler = StandardScaler()
        self.regime_labels = ['Low Vol', 'Medium Vol', 'High Vol']
        
    def extract_features(self, prices: pd.Series) -> np.ndarray:
        """Extract features for regime detection"""
        # Ensure we have a clean price series
        prices = pd.Series(prices).ffill().bfill()
        
        # Calculate returns and volatility
        returns = prices.pct_change()
        volatility = returns.rolling(window=RegimeConfig.VOLATILITY_WINDOW).std()
        
        # Calculate momentum indicators
        sma_20 = prices.rolling(window=RegimeConfig.SMA_SHORT_WINDOW).mean()
        sma_50 = prices.rolling(window=RegimeConfig.SMA_LONG_WINDOW).mean()
        momentum = (prices - sma_20) / sma_20
        trend = (sma_20 - sma_50) / sma_50
        
        # Combine features
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'momentum': momentum,
            'trend': trend
        })
        
        # Forward fill any remaining NaN values
        features = features.ffill()
        
        # If we still have NaN values at the start, fill with zeros
        features = features.fillna(0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Verify no NaN values remain
        if np.isnan(scaled_features).any():
            print("Warning: NaN values found after scaling, replacing with zeros")
            scaled_features = np.nan_to_num(scaled_features, 0)
        
        return scaled_features
    
    def fit(self, prices: pd.Series) -> None:
        """Fit HMM model to historical data"""
        try:
            # Extract features
            features = self.extract_features(prices)
            
            # Ensure we have enough data
            if len(features) < self.lookback_period:
                raise ValueError(f"Not enough data points. Need at least {self.lookback_period}")
            
            # Fit HMM
            self.hmm.fit(features)
            print("Successfully fit HMM model")
            
        except Exception as e:
            print(f"Error fitting HMM model: {str(e)}")
            raise
        
    def predict_regime(self, prices: pd.Series) -> Tuple[int, Dict[str, float]]:
        """
        Predict current market regime
        
        Returns:
            Tuple of (regime_id, regime_probabilities)
        """
        try:
            # Extract features
            features = self.extract_features(prices)
            
            if len(features) == 0:
                print("Warning: No features extracted, returning default regime")
                return 1, {label: 1.0 if i == 1 else 0.0 
                          for i, label in enumerate(self.regime_labels)}
            
            # Get regime probabilities
            regime_probs = self.hmm.predict_proba(features)
            
            # Use the last prediction
            current_probs = regime_probs[-1]
            
            # Get most likely regime
            current_regime = np.argmax(current_probs)
            
            # Create probability dictionary
            prob_dict = {
                self.regime_labels[i]: float(prob)  # Convert to native Python float
                for i, prob in enumerate(current_probs)
            }
            
            return current_regime, prob_dict
            
        except Exception as e:
            print(f"Error predicting regime: {str(e)}")
            print("Returning default regime (Medium Vol)")
            return 1, {label: 1.0 if i == 1 else 0.0 
                      for i, label in enumerate(self.regime_labels)}
    
    def get_regime_characteristics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical characteristics of each regime"""
        try:
            means = self.hmm.means_
            covars = self.hmm.covars_
            
            characteristics = {}
            for i in range(self.n_regimes):
                characteristics[self.regime_labels[i]] = {
                    'mean_return': float(means[i, 0]),  # First feature is returns
                    'volatility': float(np.sqrt(covars[i, 0, 0])),  # Diagonal element for returns
                    'transition_prob': float(self.hmm.transmat_[i, i])  # Probability of staying in regime
                }
            
            return characteristics
            
        except Exception as e:
            print(f"Error getting regime characteristics: {str(e)}")
            return {label: {
                'mean_return': 0.0,
                'volatility': 0.0,
                'transition_prob': 0.0
            } for label in self.regime_labels}
    
    def get_optimal_weights(self, regime: int, symbols: List[str]) -> Dict[str, float]:
        """Get optimal portfolio weights for given regime"""
        try:
            # Get weights for current regime
            regime_config = RegimeConfig.REGIME_WEIGHTS[regime]
            base_weights = np.array(regime_config['weights'])
            leverage = regime_config['leverage']
            
            # Adjust number of weights to match number of symbols
            if len(base_weights) < len(symbols):
                # Extend weights by equally distributing remaining weight
                remaining_weight = 1.0 - sum(base_weights)
                additional_weights = np.full(
                    len(symbols) - len(base_weights),
                    remaining_weight / (len(symbols) - len(base_weights))
                )
                weights = np.concatenate([base_weights, additional_weights])
            else:
                # Truncate weights if we have too many
                weights = base_weights[:len(symbols)]
                # Renormalize
                weights = weights / weights.sum()
            
            # Apply leverage
            weights = weights * leverage
            
            # Convert to dictionary
            return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}
            
        except Exception as e:
            print(f"Error calculating optimal weights: {str(e)}")
            # Return equal weights as fallback
            weight = 1.0 / len(symbols)
            return {symbol: weight for symbol in symbols} 