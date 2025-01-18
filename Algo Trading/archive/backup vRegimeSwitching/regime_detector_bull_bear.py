from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from config import RegimeConfig
import copy

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
        self.scaler = RobustScaler()
        self.regime_labels = ['Bull', 'Neutral', 'Bear']
        self.regime_map = None
        self.covariance_type = RegimeConfig.HMM_COVARIANCE_TYPE
        self.n_iter = RegimeConfig.HMM_N_ITER
        self.best_params = None  # Store best parameters from validation
        self.selected_features = None  # Add this to store selected feature names
    
    def _remove_correlated_features(self, features: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
        """
        Remove highly correlated features above the threshold
        
        Args:
            features: DataFrame of features
            threshold: Correlation threshold above which to remove features
            
        Returns:
            DataFrame with reduced features
        """
        try:
            # Calculate correlation matrix
            corr_matrix = features.corr().abs()
            
            # Create upper triangle mask
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features to drop
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            # Log removed features
            if to_drop:
                print(f"Removing correlated features: {to_drop}")
            
            # Drop features and return
            return features.drop(columns=to_drop)
            
        except Exception as e:
            print(f"Error in correlation removal: {e}")
            return features
    
    def _select_best_features(self, features: pd.DataFrame, target: pd.Series, 
                             n_features: int = 10) -> pd.DataFrame:
        """
        Select the most important features using mutual information
        
        Args:
            features: DataFrame of features
            target: Series of target values (returns)
            n_features: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        try:
            # Handle NaN values
            features = features.ffill().bfill()
            target = target.ffill().fillna(0)  # Fill remaining NaNs with 0
            
            # Ensure no infinite values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.ffill().fillna(0)
            
            # Initialize selector
            selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, features.shape[1]))
            
            # Fit and transform
            selected_features = selector.fit_transform(features, target)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            self.selected_features = features.columns[selected_mask].tolist()
            
            # Log feature scores
            scores = pd.Series(selector.scores_, index=features.columns)
            print("\nFeature importance scores:")
            print(scores.sort_values(ascending=False))
            
            print(f"\nSelected features: {self.selected_features}")
            
            return pd.DataFrame(selected_features, columns=self.selected_features, index=features.index)
            
        except Exception as e:
            print(f"Error in feature selection: {e}")
            raise RuntimeError(f"Feature selection failed: {e}") from e
    
    def extract_features(self, prices: pd.Series, is_training: bool = False) -> Tuple[np.ndarray, pd.Index]:
        """
        Extract features optimized for Bitcoin trend detection
        
        Args:
            prices: Price series
            is_training: Whether this is being called during training
            
        Returns:
            Tuple of (scaled features array, feature index)
        """
        try:
            print(f"Input prices length: {len(prices)}")
            
            # Get timeframes based on data frequency
            timeframes = RegimeConfig.TIMEFRAMES[RegimeConfig.DATA_FREQUENCY]
            max_lookback = RegimeConfig.get_max_lookback()
            print(f"Max lookback period: {max_lookback}")
            
            # Calculate returns
            returns = prices.pct_change()
            log_returns = np.log(prices / prices.shift(1))
            
            # 1. Bitcoin-specific Trend Features
            # Multiple timeframe moving averages (common BTC MAs)
            sma_21 = prices.rolling(timeframes['SMA_21']).mean()
            sma_50 = prices.rolling(timeframes['SMA_50']).mean()
            sma_200 = prices.rolling(timeframes['SMA_200']).mean()
            
            # Price relative to key MAs with safety checks
            price_to_sma_21 = np.where(sma_21 > 0, prices / sma_21 - 1, 0)
            price_to_sma_50 = np.where(sma_50 > 0, prices / sma_50 - 1, 0)
            price_to_sma_200 = np.where(sma_200 > 0, prices / sma_200 - 1, 0)
            
            # Moving average crossovers with safety checks
            golden_cross = np.where(sma_200 > 0, sma_50 / sma_200 - 1, 0)
            bull_cross = np.where(sma_50 > 0, sma_21 / sma_50 - 1, 0)
            
            # 2. Bitcoin Momentum Features
            # ROC with Bitcoin-specific periods
            roc_short = prices.pct_change(periods=timeframes['WEEKLY'])    # Weekly momentum
            roc_mid = prices.pct_change(periods=timeframes['MONTHLY'])     # Monthly momentum
            roc_long = prices.pct_change(periods=timeframes['QUARTERLY'])  # Quarterly momentum
            
            # Cumulative returns for trend strength
            cum_returns_30 = (1 + returns).rolling(window=timeframes['MONTHLY']).apply(lambda x: x.prod()) - 1
            
            # 3. Bitcoin Volatility Features
            # Volatility in different regimes
            vol_short = returns.rolling(timeframes['WEEKLY']).std()     # Weekly vol
            vol_mid = returns.rolling(timeframes['MONTHLY']).std()      # Monthly vol
            vol_long = returns.rolling(timeframes['QUARTERLY']).std()   # Quarterly vol
            
            # Volatility ratios with safety checks
            vol_ratio_short = np.where(vol_mid > 0, vol_short / vol_mid, 1)   # Vol regime change
            vol_ratio_long = np.where(vol_long > 0, vol_mid / vol_long, 1)    # Long-term vol trend
            
            # 4. Market Structure Features
            # Higher highs and lower lows (key for BTC trends)
            rolling_max_30 = prices.rolling(timeframes['MONTHLY']).max()
            rolling_min_30 = prices.rolling(timeframes['MONTHLY']).min()
            
            # Price position within range with safety check
            price_range = rolling_max_30 - rolling_min_30
            price_position = np.where(price_range > 0, 
                                    (prices - rolling_min_30) / price_range,
                                    0.5)  # Default to middle when no range
            
            # New highs/lows frequency
            new_highs = (prices == rolling_max_30).rolling(timeframes['MONTHLY']).sum() / timeframes['MONTHLY']
            new_lows = (prices == rolling_min_30).rolling(timeframes['MONTHLY']).sum() / timeframes['MONTHLY']
            
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
            
            # Add more discriminative features
            features['volatility_regime'] = vol_mid / vol_mid.rolling(timeframes['MONTHLY']).mean()
            features['trend_strength'] = abs(price_to_sma_50)
            features['momentum_regime'] = (
                prices.pct_change(timeframes['WEEKLY']).rolling(timeframes['MONTHLY']).mean() /
                prices.pct_change(timeframes['WEEKLY']).rolling(timeframes['MONTHLY']).std()
            )
            
            # Calculate returns for feature selection
            # Align returns with features by using the same index
            returns = prices.pct_change()
            
            # Handle NaN values before feature selection
            features = features.ffill().bfill()
            returns = returns.ffill().fillna(0)
            
            # Ensure features and returns have the same index
            common_index = features.index.intersection(returns.index)
            features = features.loc[common_index]
            returns = returns.loc[common_index]
            
            # Remove highly correlated features
            features = self._remove_correlated_features(features)
            
            # Only perform feature selection during training
            if is_training:
                features = self._select_best_features(
                    features, 
                    returns,
                    n_features=RegimeConfig.N_FEATURES
                )
            else:
                # Use previously selected features
                if self.selected_features is None:
                    raise RuntimeError("Model must be trained before extracting features for prediction")
                features = features[self.selected_features]
            
            # Store the index before scaling
            feature_index = features.index
            
            # Standardize features with robust scaling
            if not hasattr(self.scaler, 'center_'):
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = self.scaler.transform(features)
            
            # Add feature importance check
            print("Feature variances:")
            for col, var in zip(features.columns, np.var(features_scaled, axis=0)):
                print(f"{col}: {var:.6f}")
            
            return features_scaled, feature_index
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            raise RuntimeError(f"Failed to extract features: {e}")
    
    def fit(self, train_prices: pd.Series, val_prices: pd.Series = None) -> None:
        """Fit HMM model to training data and optionally tune parameters using validation data"""
        try:
            # Extract features with feature selection (training mode)
            features, feature_index = self.extract_features(train_prices, is_training=True)
            print(f"Training features shape: {features.shape}")
            
            # Add feature variation check
            feature_std = np.std(features, axis=0)
            low_var_features = np.where(feature_std < 1e-6)[0]
            if len(low_var_features) > 0:
                print(f"Warning: Low variance in features: {low_var_features}")
            
            # Initialize HMM without any automatic initialization
            self.hmm = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=42,
                init_params='',    
                params='stmc'      
            )
            
            # Set non-zero initial probabilities
            self.hmm.startprob_ = np.ones(self.n_regimes) / self.n_regimes
            
            # Set non-zero transition matrix with prior knowledge
            self.hmm.transmat_ = np.array([
                [0.95, 0.04, 0.01],  
                [0.03, 0.94, 0.03],  
                [0.01, 0.04, 0.95]   
            ])
            
            # Initialize diagonal covariances with ones
            n_features = features.shape[1]
            self.hmm.covars_ = np.ones((self.n_regimes, n_features))
            
            # Fit model with multiple restarts
            best_score = float('-inf')
            best_hmm = None
            
            for _ in range(5):  # Try 5 different initializations
                try:
                    self.hmm.fit(features)
                    score = self.hmm.score(features)
                    if score > best_score:
                        best_score = score
                        best_hmm = copy.deepcopy(self.hmm)
                except Exception as e:
                    print(f"Warning: HMM fitting attempt failed: {e}")
                    continue
            
            if best_hmm is not None:
                self.hmm = best_hmm
            else:
                raise ValueError("All HMM fitting attempts failed")
            
            # Get regime predictions and ensure all states are used
            regimes = self.hmm.predict(features)
            unique_regimes = np.unique(regimes)
            if len(unique_regimes) < self.n_regimes:
                print("Warning: Not all regimes were detected. Adjusting parameters...")
                # Adjust transition matrix to encourage exploration
                self.hmm.transmat_ = self.hmm.transmat_ * 0.9 + 0.1 / self.n_regimes
                regimes = self.hmm.predict(features)
            
            # Calculate returns only for the period where we have features
            returns = train_prices.pct_change()
            returns = returns.loc[feature_index]  # Now using the stored feature_index
            
            # Sort regimes by return characteristics
            regime_stats = []
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