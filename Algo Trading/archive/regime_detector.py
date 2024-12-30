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
        self.hmm = None  # Initialize in fit() to allow for different params
        self.scaler = StandardScaler()
        self.regime_labels = ['Low Vol', 'Medium Vol', 'High Vol']
        self.best_params = None
        self.best_score = float('inf')
        
    def extract_features(self, prices: pd.Series) -> np.ndarray:
        """Extract features for regime detection with enhanced trend sensitivity"""
        try:
            # Calculate returns and absolute returns
            returns = prices.pct_change()
            abs_returns = np.abs(returns)
            
            # 1. Volatility features
            vol_st = returns.rolling(RegimeConfig.VOLATILITY_WINDOW).std()
            vol_lt = returns.rolling(self.lookback_period).std()
            vol_ratio = vol_st / vol_lt
            
            # 2. Enhanced trend features
            # Multiple timeframe moving averages
            sma_short = prices.rolling(RegimeConfig.SMA_SHORT_WINDOW).mean()
            sma_mid = prices.rolling(RegimeConfig.SMA_MID_WINDOW).mean()
            sma_long = prices.rolling(RegimeConfig.SMA_LONG_WINDOW).mean()
            
            # Trend strength indicators
            trend_st = (prices - sma_short) / sma_short
            trend_mt = (prices - sma_mid) / sma_mid
            trend_lt = (prices - sma_long) / sma_long
            
            # Trend consistency (alignment of different timeframes)
            trend_alignment = np.sign(trend_st) + np.sign(trend_mt) + np.sign(trend_lt)
            
            # 3. Downtrend specific features
            # Consecutive down days
            down_streak = returns.rolling(10).apply(
                lambda x: (x < 0).sum() / len(x)
            )
            
            # Lower lows and lower highs
            rolling_max = prices.rolling(RegimeConfig.LOOKBACK_PERIOD).max()
            rolling_min = prices.rolling(RegimeConfig.LOOKBACK_PERIOD).min()
            price_position = (prices - rolling_min) / (rolling_max - rolling_min)
            
            # Downside volatility
            down_vol = returns.where(returns < 0, 0).rolling(RegimeConfig.VOLATILITY_WINDOW).std()
            up_vol = returns.where(returns > 0, 0).rolling(RegimeConfig.VOLATILITY_WINDOW).std()
            vol_skew = down_vol / up_vol
            
            # 4. Market stress indicators
            # Drawdown from peak
            drawdown = (prices - rolling_max) / rolling_max
            
            # Return distribution features
            skew = returns.rolling(self.lookback_period).skew()
            kurt = returns.rolling(self.lookback_period).kurt()
            
            # Combine all features
            features = pd.DataFrame({
                'vol_st': vol_st,
                'vol_lt': vol_lt,
                'vol_ratio': vol_ratio,
                'trend_st': trend_st,
                'trend_mt': trend_mt,
                'trend_lt': trend_lt,
                'trend_alignment': trend_alignment,
                'down_streak': down_streak,
                'price_position': price_position,
                'vol_skew': vol_skew,
                'drawdown': drawdown,
                'skew': skew,
                'kurt': kurt
            })
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Handle NaN values
            features_scaled = np.nan_to_num(features_scaled, nan=0.0)
            
            return features_scaled
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return np.zeros((len(prices), 13))  # Return zero features as fallback
    
    def create_hmm(self, covariance_type: str = "full", n_iter: int = 100) -> hmm.GaussianHMM:
        """Create HMM model with given parameters"""
        return hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42
        )
    
    def evaluate_model(self, model: hmm.GaussianHMM, features: np.ndarray, val_features: np.ndarray) -> float:
        """Evaluate model performance using validation data"""
        try:
            # Calculate log likelihood on validation set
            val_score = model.score(val_features)
            
            # Calculate regime distribution
            regimes = model.predict(features)
            regime_counts = np.bincount(regimes, minlength=self.n_regimes)
            regime_props = regime_counts / len(regimes)
            
            # Penalize if any regime has too low or too high proportion
            balance_penalty = 0
            for prop in regime_props:
                if prop < 0.1:  # Too few samples in regime
                    balance_penalty += abs(0.1 - prop) * 100
                elif prop > 0.6:  # Too many samples in regime
                    balance_penalty += abs(prop - 0.6) * 100
            
            # Combine log likelihood with balance penalty
            score = -val_score + balance_penalty
            return score
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return float('inf')
    
    def tune_hyperparameters(self, train_features: np.ndarray, val_features: np.ndarray) -> Dict:
        """Tune HMM hyperparameters using validation data"""
        covariance_types = ['full', 'diag', 'spherical']
        n_iters = [100, 200, 300]
        
        best_score = float('inf')
        best_params = None
        
        for cov_type in covariance_types:
            for n_iter in n_iters:
                try:
                    # Create and fit model
                    model = self.create_hmm(covariance_type=cov_type, n_iter=n_iter)
                    model.fit(train_features)
                    
                    # Evaluate model
                    score = self.evaluate_model(model, train_features, val_features)
                    
                    print(f"Params: {cov_type}, {n_iter} - Score: {score:.2f}")
                    
                    if score < best_score:
                        best_score = score
                        best_params = {'covariance_type': cov_type, 'n_iter': n_iter}
                        
                except Exception as e:
                    print(f"Error with params {cov_type}, {n_iter}: {str(e)}")
                    continue
        
        return best_params
    
    def fit(self, train_prices: pd.Series, val_prices: pd.Series = None) -> None:
        """Fit HMM model to training data"""
        try:
            # Extract features
            features = self.extract_features(train_prices)
            
            # Initialize HMM
            self.hmm = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=42
            )
            
            # Fit model
            self.hmm.fit(features)
            
            # Get regime predictions
            regimes = self.hmm.predict(features)
            
            # Sort regimes by volatility to ensure consistent labeling
            regime_vols = []
            for i in range(self.n_regimes):
                regime_mask = (regimes == i)
                if np.any(regime_mask):
                    regime_vol = np.std(train_prices.pct_change()[regime_mask])
                    regime_vols.append((i, regime_vol))
            
            # Sort regimes by volatility (ascending)
            regime_vols.sort(key=lambda x: x[1])
            
            # Create mapping from old to new labels
            self.regime_map = {old: new for new, (old, _) in enumerate(regime_vols)}
            
            # Validate on validation set if provided
            if val_prices is not None:
                val_features = self.extract_features(val_prices)
                val_regimes = self.hmm.predict(val_features)
                
                # Map regimes using volatility-based mapping
                val_regimes = np.array([self.regime_map[r] for r in val_regimes])
                
                # Print validation set regime distribution
                regime_counts = np.bincount(val_regimes, minlength=self.n_regimes)
                print("\nValidation Set Regime Distribution:")
                for i, count in enumerate(regime_counts):
                    pct = count / len(val_regimes) * 100
                    print(f"{self.regime_labels[i]}: {pct:.1f}%")
            
        except Exception as e:
            print(f"Error fitting regime detector: {str(e)}")
    
    def predict_regime(self, prices: pd.Series) -> Tuple[int, np.ndarray]:
        """Predict current market regime"""
        try:
            # Extract features
            features = self.extract_features(prices)
            
            # Get regime probabilities
            probs = self.hmm.predict_proba(features)
            
            # Get most likely regime
            regime = self.hmm.predict(features)[-1]
            
            # Map regime using volatility-based mapping
            regime = self.regime_map[regime]
            
            return regime, probs[-1]
            
        except Exception as e:
            print(f"Error predicting regime: {str(e)}")
            return 1, np.array([0.0, 1.0, 0.0])  # Default to medium volatility
    
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