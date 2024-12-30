from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from config import RegimeConfig

class RegimeDetectorBullBear:
    """Detect market regimes (Bull/Bear) using Hidden Markov Models"""
    
    def __init__(self, n_regimes: int = 3,  # Changed to 3 regimes: Bull, Neutral, Bear
                 lookback_period: int = RegimeConfig.LOOKBACK_PERIOD):
        """
        Initialize bull/neutral/bear regime detector
        
        Args:
            n_regimes: Number of market regimes (fixed to 3: Bull, Neutral, Bear)
            lookback_period: Number of periods to use for regime detection
        """
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.hmm = None
        self.scaler = StandardScaler()
        self.regime_labels = ['Bull', 'Neutral', 'Bear']
        self.regime_map = None
        self.covariance_type = RegimeConfig.HMM_COVARIANCE_TYPE
        self.n_iter = RegimeConfig.HMM_N_ITER
        self.best_params = None  # Store best parameters from validation
    
    def extract_features(self, prices: pd.Series) -> np.ndarray:
        """Extract features optimized for Bitcoin trend detection"""
        try:
            # Calculate returns
            returns = prices.pct_change()
            log_returns = np.log(prices / prices.shift(1))
            
            # 1. Bitcoin-specific Trend Features
            # Multiple timeframe moving averages (common BTC MAs)
            sma_21 = prices.rolling(21).mean()  # Common BTC trading MA
            sma_50 = prices.rolling(50).mean()  # Medium-term trend
            sma_200 = prices.rolling(200).mean()  # Long-term trend
            
            # Price relative to key MAs with safety checks
            price_to_sma_21 = np.where(sma_21 > 0, prices / sma_21 - 1, 0)
            price_to_sma_50 = np.where(sma_50 > 0, prices / sma_50 - 1, 0)
            price_to_sma_200 = np.where(sma_200 > 0, prices / sma_200 - 1, 0)
            
            # Moving average crossovers with safety checks
            golden_cross = np.where(sma_200 > 0, sma_50 / sma_200 - 1, 0)
            bull_cross = np.where(sma_50 > 0, sma_21 / sma_50 - 1, 0)
            
            # 2. Bitcoin Momentum Features
            # ROC with Bitcoin-specific periods
            roc_short = prices.pct_change(periods=7)   # Weekly momentum
            roc_mid = prices.pct_change(periods=30)    # Monthly momentum
            roc_long = prices.pct_change(periods=90)   # Quarterly momentum
            
            # Cumulative returns for trend strength
            cum_returns_30 = (1 + returns).rolling(window=30).apply(lambda x: x.prod()) - 1
            
            # 3. Bitcoin Volatility Features
            # Volatility in different regimes
            vol_short = returns.rolling(7).std()    # Weekly vol
            vol_mid = returns.rolling(30).std()     # Monthly vol
            vol_long = returns.rolling(90).std()    # Quarterly vol
            
            # Volatility ratios with safety checks
            vol_ratio_short = np.where(vol_mid > 0, vol_short / vol_mid, 1)   # Vol regime change
            vol_ratio_long = np.where(vol_long > 0, vol_mid / vol_long, 1)    # Long-term vol trend
            
            # 4. Market Structure Features
            # Higher highs and lower lows (key for BTC trends)
            rolling_max_30 = prices.rolling(30).max()
            rolling_min_30 = prices.rolling(30).min()
            
            # Price position within range with safety check
            price_range = rolling_max_30 - rolling_min_30
            price_position = np.where(price_range > 0, 
                                    (prices - rolling_min_30) / price_range,
                                    0.5)  # Default to middle when no range
            
            # New highs/lows frequency
            new_highs = (prices == rolling_max_30).rolling(30).sum() / 30
            new_lows = (prices == rolling_min_30).rolling(30).sum() / 30
            
            # 5. Bitcoin-specific Risk Metrics
            # Drawdown from ATH with safety check
            rolling_max_all = prices.expanding().max()
            drawdown = np.where(rolling_max_all > 0,
                               (prices - rolling_max_all) / rolling_max_all,
                               0)
            
            # NVTS-like feature with safety check
            vol_price_product = vol_mid * prices
            nvts = np.where(vol_price_product > 0,
                           prices / vol_price_product,
                           1)  # Default to 1 when undefined
            
            # Combine features
            features = pd.DataFrame({
                # Trend features
                'price_to_sma_21': price_to_sma_21,
                'price_to_sma_50': price_to_sma_50,
                'price_to_sma_200': price_to_sma_200,
                'golden_cross': golden_cross,
                'bull_cross': bull_cross,
                
                # Momentum features
                'roc_short': roc_short,
                'roc_mid': roc_mid,
                'roc_long': roc_long,
                'cum_returns_30': cum_returns_30,
                
                # Volatility features
                'vol_ratio_short': vol_ratio_short,
                'vol_ratio_long': vol_ratio_long,
                
                # Market structure features
                'price_position': price_position,
                'new_highs': new_highs,
                'new_lows': new_lows,
                'drawdown': drawdown,
                'nvts': nvts
            })
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Handle NaN values
            features_scaled = np.nan_to_num(features_scaled, nan=0.0)
            
            return features_scaled
            
        except Exception:
            return np.zeros((len(prices), 16))  # Return zero features as fallback
    
    def fit(self, train_prices: pd.Series, val_prices: pd.Series = None) -> None:
        """Fit HMM model to training data and optionally tune parameters using validation data"""
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
            
            # Sort regimes by return characteristics
            regime_stats = []
            returns = train_prices.pct_change()
            for i in range(self.n_regimes):
                regime_mask = (regimes == i)
                if np.any(regime_mask):
                    regime_returns = returns[regime_mask]
                    stats = {
                        'regime': i,
                        'mean_return': np.mean(regime_returns),
                        'volatility': np.std(regime_returns),
                        'sharpe': np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0
                    }
                    regime_stats.append(stats)
            
            # Sort regimes by Sharpe ratio (descending)
            regime_stats.sort(key=lambda x: x['sharpe'], reverse=True)
            
            # Map regimes (highest Sharpe = Bull, middle = Neutral, lowest = Bear)
            self.regime_map = {stats['regime']: i for i, stats in enumerate(regime_stats)}
            
            # If validation data is provided, tune parameters
            if val_prices is not None:
                self._tune_parameters(val_prices)
            
        except Exception:
            raise
    
    def _tune_parameters(self, val_prices: pd.Series) -> None:
        """Tune strategy parameters using validation data"""
        try:
            # Get regime predictions for validation set
            val_features = self.extract_features(val_prices)
            val_regimes = self.hmm.predict(val_features)
            val_regimes = np.array([self.regime_map[r] for r in val_regimes])
            val_returns = val_prices.pct_change().fillna(0)  # Fill NaN with 0 for first return
            
            best_params = {}
            best_score = float('-inf')
            
            # Grid search over parameter ranges
            for leverage_bull in np.arange(*RegimeConfig.VALIDATION_RANGES['leverage'][0], 0.1):
                for leverage_neutral in np.arange(*RegimeConfig.VALIDATION_RANGES['leverage'][1], 0.1):
                    for leverage_bear in np.arange(*RegimeConfig.VALIDATION_RANGES['leverage'][2], 0.1):
                        # Create parameter set
                        params = {
                            'leverage': {
                                0: leverage_bull,
                                1: leverage_neutral,
                                2: leverage_bear
                            }
                        }
                        
                        # Calculate regime-specific returns
                        regime_returns = []
                        for i in range(self.n_regimes):
                            regime_mask = (val_regimes == i)
                            if np.any(regime_mask):  # Only process if regime exists
                                regime_rets = val_returns[regime_mask] * params['leverage'][i]
                                regime_returns.extend(regime_rets)
                        
                        # Convert to numpy array for calculations
                        regime_returns = np.array(regime_returns)
                        
                        # Calculate performance metrics with safety checks
                        if len(regime_returns) > 0:
                            returns_std = np.std(regime_returns)
                            downside_returns = np.minimum(regime_returns, 0)
                            downside_std = np.std(downside_returns)
                            
                            sharpe = np.mean(regime_returns) / returns_std if returns_std > 0 else 0
                            sortino = np.mean(regime_returns) / downside_std if downside_std > 0 else 0
                            
                            # Calculate score (weighted combination of metrics)
                            score = RegimeConfig.VALIDATION_METRICS['sharpe_ratio'] * sharpe + \
                                   RegimeConfig.VALIDATION_METRICS['sortino_ratio'] * sortino
                            
                            # Update best parameters if score is better
                            if score > best_score and not np.isnan(score):
                                best_score = score
                                best_params = params.copy()
            
            # Store best parameters
            if not best_params:
                raise ValueError("Failed to find valid parameters during validation")
            self.best_params = best_params
            
        except Exception:
            raise
    
    def get_regime_characteristics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical characteristics of each regime"""
        try:
            means = self.hmm.means_
            covars = self.hmm.covars_
            
            characteristics = {}
            for i in range(self.n_regimes):
                mapped_i = self.regime_map[i]
                characteristics[self.regime_labels[mapped_i]] = {
                    'mean_return': float(means[i, 4]),  # ROC short feature
                    'trend_strength': float(means[i, 8]),  # Trend strength feature
                    'transition_prob': float(self.hmm.transmat_[i, i])  # Probability of staying in regime
                }
            
            return characteristics
            
        except Exception:
            return {label: {
                'mean_return': 0.0,
                'trend_strength': 0.0,
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
                remaining_weight = 1.0 - sum(base_weights)
                additional_weights = np.full(
                    len(symbols) - len(base_weights),
                    remaining_weight / (len(symbols) - len(base_weights))
                )
                weights = np.concatenate([base_weights, additional_weights])
            else:
                weights = base_weights[:len(symbols)]
                weights = weights / weights.sum()
            
            # Apply leverage
            weights = weights * leverage
            
            return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}
            
        except Exception:
            weight = 1.0 / len(symbols)
            return {symbol: weight for symbol in symbols} 