import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from typing import List, Dict, Tuple
from portfolio import Portfolio
from config import (SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, 
                   DROPOUT_RATE, LSTM_UNITS, DENSE_UNITS, EARLY_STOPPING_PATIENCE,
                   REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, TECHNICAL_INDICATORS,
                   TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, TEST_MODE)

class WeightNormalization(tf.keras.layers.Layer):
    """Custom layer to normalize weights to sum to 1"""
    def call(self, inputs):
        return inputs / tf.keras.backend.sum(inputs, axis=1, keepdims=True)

@tf.keras.utils.register_keras_serializable()
class ReturnMaximizationLoss(tf.keras.losses.Loss):
    """Custom loss function for return maximization"""
    
    def __init__(self, name='return_maximization_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
    
    def get_config(self):
        config = super().get_config()
        return config
    
    def call(self, y_true, y_pred):
        # Ensure predictions are in valid range
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Check if shapes match
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        
        tf.debugging.assert_equal(
            y_true_shape,
            y_pred_shape,
            message="Shape mismatch between y_true and y_pred"
        )
        
        # Normalize predictions to sum to 1
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=1, keepdims=True)
        
        # Mean squared error (to maintain some accuracy)
        mse = 0.1 * tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Portfolio return (negative because we want to maximize it)
        returns = -tf.reduce_mean(y_pred * y_true)
        
        # Diversification penalty (reduced weight compared to returns)
        concentration = tf.reduce_sum(tf.square(y_pred), axis=1)
        diversification_penalty = 0.05 * tf.reduce_mean(concentration)
        
        # Combine losses with emphasis on returns
        total_loss = returns + mse + diversification_penalty
        
        # Handle NaN/Inf values
        return tf.where(
            tf.math.is_finite(total_loss),
            total_loss,
            tf.constant(1e3, dtype=tf.float32)
        )

@tf.keras.utils.register_keras_serializable()
class StableMAE(tf.keras.metrics.Mean):
    """Custom MAE metric with stability measures"""
    
    def __init__(self, name='stable_mae', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure predictions are in valid range
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Normalize predictions to sum to 1
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=1, keepdims=True)
        
        # Calculate MAE with clipping for stability
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        mae = tf.where(tf.math.is_finite(mae), mae, tf.constant(1.0, dtype=tf.float32))
        
        return super().update_state(mae, sample_weight=sample_weight)

class CryptoLSTM:
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.history = None
        self.training = False
        self.outlier_threshold = 3
        
    def create_model(self, n_features: int, n_assets: int):
        """Create simplified LSTM model with focus on stability"""
        # Calculate total features after reshaping
        total_features = n_features * n_assets
        input_shape = (self.sequence_length, total_features)
        
        print(f"\nCreating model with:")
        print(f"- Input shape: {input_shape}")
        print(f"- Number of assets: {n_assets}")
        print(f"- Features per asset: {n_features}")
        print(f"- Total features: {total_features}")
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),
            
            # First LSTM layer with batch normalization after LSTM
            tf.keras.layers.LSTM(
                units=total_features,  # Match input feature dimension
                return_sequences=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Dropout(0.2),
            
            # Second LSTM layer with batch normalization after LSTM
            tf.keras.layers.LSTM(
                units=total_features // 2,  # Reduce dimension gradually
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Dropout(0.2),
            
            # Dense layers
            tf.keras.layers.Dense(total_features // 4, activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Dropout(0.2),
            
            # Output layer with dynamic size based on number of assets
            tf.keras.layers.Dense(n_assets, activation='softmax')
        ])
        
        # Use a smaller learning rate
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,  # Reduced learning rate
            clipnorm=1.0,        # Gradient clipping
            epsilon=1e-7         # Numerical stability
        )
        
        model.compile(
            optimizer=optimizer,
            loss=ReturnMaximizationLoss(),
            metrics=[StableMAE()]  # Use our custom StableMAE metric
        )
        
        # Print model summary
        model.summary()
        
        self.model = model
        return model

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets"""
        n = len(X)
        train_end = int(n * TRAIN_SIZE)
        val_end = train_end + int(n * VALIDATION_SIZE)
        
        # Training set
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        # Validation set
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        # Test set
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, data: pd.DataFrame) -> Dict:
        """Train the model with enhanced stability measures"""
        self.training = True
        
        X, y = self.prepare_data(data)
        
        if len(X.shape) != 4:  # [samples, sequence_length, n_assets, n_features]
            raise ValueError(f"Expected X to have 4 dimensions, got {len(X.shape)}")
        
        # Get dimensions
        n_samples, seq_len, n_assets, n_features = X.shape
        total_features = n_assets * n_features
        
        # Create model if it doesn't exist
        if self.model is None:
            self.create_model(n_features=n_features, n_assets=n_assets)
        
        # Reshape X to [samples, sequence_length, n_features * n_assets]
        X = X.reshape(n_samples, seq_len, total_features)
        
        # Ensure y is 2D array
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Ensure targets are normalized and handle any NaN values
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        row_sums = np.sum(y, axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        y = y / row_sums
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Verify shapes
        print(f"Training shapes - X: {X_train.shape}, y: {y_train.shape}")
        print(f"Input features per timestep: {total_features}")
        print(f"Output dimension: {y_train.shape[1]}")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                min_delta=1e-4
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train with fixed batch size for stability
        batch_size = min(32, len(X_train))
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
        
        self.model = tf.keras.models.load_model(
            'best_model.keras',
            custom_objects={
                'ReturnMaximizationLoss': ReturnMaximizationLoss,
                'StableMAE': StableMAE
            }
        )
        
        # Evaluate
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        self.training = False
        
        results = {
            'train_loss': self.history.history['loss'][-1],
            'val_loss': self.history.history['val_loss'][-1],
            'test_loss': test_loss[0],
            'train_mae': self.history.history['stable_mae'][-1],
            'val_mae': self.history.history['val_stable_mae'][-1],
            'test_mae': test_loss[1]
        }
        
        return results

    def calculate_rsi(self, returns: List[float], period: int = TECHNICAL_INDICATORS['RSI_PERIOD']) -> float:
        """Calculate Relative Strength Index"""
        if len(returns) < period:
            return 50
        
        gains = []
        losses = []
        for r in returns[-period:]:
            if r > 0:
                gains.append(r)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(r))
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: List[float]) -> Tuple[float, float]:
        """Calculate MACD and Signal line"""
        if len(prices) < TECHNICAL_INDICATORS['MACD_SLOW']:
            return 0, 0
        
        # Calculate EMAs
        ema_fast = pd.Series(prices).ewm(
            span=TECHNICAL_INDICATORS['MACD_FAST'], adjust=False).mean().iloc[-1]
        ema_slow = pd.Series(prices).ewm(
            span=TECHNICAL_INDICATORS['MACD_SLOW'], adjust=False).mean().iloc[-1]
        macd = ema_fast - ema_slow
        
        # Calculate Signal line
        macd_series = pd.Series(prices).ewm(
            span=TECHNICAL_INDICATORS['MACD_FAST'], adjust=False).mean() - \
            pd.Series(prices).ewm(
                span=TECHNICAL_INDICATORS['MACD_SLOW'], adjust=False).mean()
        signal = macd_series.ewm(
            span=TECHNICAL_INDICATORS['MACD_SIGNAL'], adjust=False).mean().iloc[-1]
        
        return macd, signal

    def calculate_bollinger_bands(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        window = TECHNICAL_INDICATORS['BB_PERIOD']
        if len(prices) < window:
            return 0, 0, 0
        
        prices_series = pd.Series(prices[-window:])
        sma = prices_series.mean()
        std = prices_series.std()
        
        upper_band = sma + (TECHNICAL_INDICATORS['BB_STD'] * std)
        lower_band = sma - (TECHNICAL_INDICATORS['BB_STD'] * std)
        
        return upper_band, sma, lower_band

    def extract_features(self, portfolio: Portfolio, 
                        current_prices: Dict[str, dict], 
                        lookback: int = None) -> pd.DataFrame:
        """Extract enhanced features for prediction"""
        # Use sequence_length as the lookback period
        if lookback is None:
            lookback = self.sequence_length
        
        features = []
        total_assets = len(portfolio.portfolio_df)
        min_required_data = self.sequence_length  # Need sequence_length for prediction
        
        for i, (symbol, row) in enumerate(portfolio.portfolio_df.iterrows()):
            try:
                asset = row['asset']
                returns = list(asset.get_return())
                prices = list(asset.prices)
                
                # If we don't have enough data, use forward fill to extend the data
                while len(returns) < min_required_data:
                    returns.append(returns[-1] if returns else 0)
                while len(prices) < min_required_data:
                    prices.append(prices[-1] if prices else current_prices[symbol]['markPrice'])
                    
                current_price = current_prices.get(symbol, {}).get('markPrice')
                if current_price is None:
                    continue
                
                # Use all available data points up to lookback
                actual_lookback = min(lookback, len(prices))
                
                # Basic features
                price_sma = np.mean(prices[-actual_lookback:])
                price_std = np.std(prices[-actual_lookback:])
                volatility = np.std(returns[-actual_lookback:])
                momentum = sum(returns[-actual_lookback:])
                
                # Calculate technical indicators
                rsi = self.calculate_rsi(returns)
                macd, macd_signal = self.calculate_macd(prices)
                upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(prices)
                bb_position = (current_price - lower_bb) / (upper_bb - lower_bb) if upper_bb != lower_bb else 0.5
                
                price_to_sma = current_price / price_sma if price_sma != 0 else 1
                
                # Create multiple feature rows for the sequence
                for j in range(min_required_data):
                    # Calculate statistical moments
                    returns_array = np.array(returns[-actual_lookback:])
                    skewness = 0.0 if len(returns_array) < 3 else pd.Series(returns_array).skew()
                    kurtosis = 0.0 if len(returns_array) < 4 else pd.Series(returns_array).kurt()
                    
                    # Calculate price changes
                    price_changes = {}
                    for horizon in [1, 3, 5, 10]:
                        if len(prices) > horizon:
                            change = (prices[-1] - prices[-horizon-1]) / prices[-horizon-1]
                        else:
                            change = 0.0
                        price_changes[f'price_change_{horizon}'] = change
                    
                    # Calculate additional indicators
                    macd_hist = macd - macd_signal
                    rsi_prev = self.calculate_rsi(returns[:-1]) if len(returns) > 1 else rsi
                    rsi_change = rsi - rsi_prev
                    
                    # Volatility ratios
                    vol_5 = np.std(returns[-5:]) if len(returns) >= 5 else volatility
                    vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else volatility
                    vol_ratio_5_20 = vol_5 / vol_20 if vol_20 != 0 else 1.0
                    
                    # Basic features first (in same order as prepare_data)
                    feature_dict = {
                        'return': returns[-(j+1)] if j < len(returns) else returns[-1],
                        'volatility': volatility,
                        'momentum': momentum,
                        'price': prices[-(j+1)] if j < len(prices) else prices[-1],
                        'price_to_sma': price_to_sma,
                        'price_std': price_std,
                        'rsi': rsi,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'bb_position': bb_position,
                        # Statistical moments
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        # Price changes
                        'price_change_1': price_changes['price_change_1'],
                        'price_change_3': price_changes['price_change_3'],
                        'price_change_5': price_changes['price_change_5'],
                        'price_change_10': price_changes['price_change_10'],
                        # Technical features
                        'vol_ratio_5_20': vol_ratio_5_20,
                        'rsi_change': rsi_change,
                        'macd_hist': macd_hist,
                        # Rank features
                        'return_rank': 0.5,
                        'volatility_rank': 0.5,
                        'momentum_rank': 0.5,
                        'rsi_rank': 0.5
                    }
                    features.append(feature_dict)
                
            except Exception:
                # If there's an error processing this asset, fill with default values
                default_price = current_prices.get(symbol, {}).get('markPrice', 0)
                for j in range(min_required_data):
                    feature_dict = {
                        # Basic features
                        'return': 0,
                        'volatility': 0,
                        'momentum': 0,
                        'price': default_price,
                        'price_to_sma': 1,
                        'price_std': 0,
                        'rsi': 50,
                        'macd': 0,
                        'macd_signal': 0,
                        'bb_position': 0.5,
                        # Statistical moments
                        'skewness': 0,
                        'kurtosis': 0,
                        # Price changes
                        'price_change_1': 0,
                        'price_change_3': 0,
                        'price_change_5': 0,
                        'price_change_10': 0,
                        # Technical features
                        'vol_ratio_5_20': 1,
                        'rsi_change': 0,
                        'macd_hist': 0,
                        # Rank features
                        'return_rank': 0.5,
                        'volatility_rank': 0.5,
                        'momentum_rank': 0.5,
                        'rsi_rank': 0.5
                    }
                    features.append(feature_dict)

        features_df = pd.DataFrame(features)
        
        # Verify we have enough data points for each asset
        expected_rows = total_assets * min_required_data
        if len(features_df) < expected_rows:
            # Fill missing rows with default values
            missing_rows = expected_rows - len(features_df)
            default_rows = []
            for _ in range(missing_rows):
                default_rows.append({
                    # Basic features
                    'return': 0,
                    'volatility': 0,
                    'momentum': 0,
                    'price': 0,
                    'price_to_sma': 1,
                    'price_std': 0,
                    'rsi': 50,
                    'macd': 0,
                    'macd_signal': 0,
                    'bb_position': 0.5,
                    # Statistical moments
                    'skewness': 0,
                    'kurtosis': 0,
                    # Price changes
                    'price_change_1': 0,
                    'price_change_3': 0,
                    'price_change_5': 0,
                    'price_change_10': 0,
                    # Technical features
                    'vol_ratio_5_20': 1,
                    'rsi_change': 0,
                    'macd_hist': 0,
                    # Rank features
                    'return_rank': 0.5,
                    'volatility_rank': 0.5,
                    'momentum_rank': 0.5,
                    'rsi_rank': 0.5
                })
            features_df = pd.concat([features_df, pd.DataFrame(default_rows)], ignore_index=True)
        
        # Add asset index after scaling
        features_df['asset_index'] = np.tile(np.arange(total_assets), len(features_df) // total_assets)
            
        return features_df

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM with forward-looking targets"""
        if data.empty:
            raise ValueError("Input data is empty. Check feature extraction.")
        
        # Get unique timestamps and assets first
        if isinstance(data.index, pd.MultiIndex):
            # For multi-index (timestamp, symbol)
            unique_timestamps = data.index.get_level_values(0).unique()
            n_assets = len(data.index.get_level_values(1).unique())
        else:
            # For single index, assume data is stacked by asset for each timestamp
            unique_timestamps = data.index.unique()
            # Try to determine n_assets from the data structure
            if 'asset_index' in data.columns:
                n_assets = len(data['asset_index'].unique())
            else:
                # Try to infer from the data structure
                sample_size = len(data.loc[unique_timestamps[0]:unique_timestamps[0]])
                if sample_size > 0:
                    n_assets = sample_size
                else:
                    raise ValueError("Cannot determine number of assets from data structure")
        
        n_timestamps_original = len(unique_timestamps)
        
        # Filter timestamps that don't have all assets
        valid_timestamps = []
        for ts in unique_timestamps:
            if isinstance(data.index, pd.MultiIndex):
                assets_at_ts = len(data.loc[ts])
            else:
                # For single index, check if we have n_assets rows for this timestamp
                start_idx = list(unique_timestamps).index(ts) * n_assets
                end_idx = start_idx + n_assets
                if end_idx <= len(data):
                    assets_at_ts = n_assets
                else:
                    assets_at_ts = 0
            
            if assets_at_ts == n_assets:
                valid_timestamps.append(ts)
        
        n_timestamps = len(valid_timestamps)
        
        if n_timestamps <= self.sequence_length:
            raise ValueError(f"Not enough valid timestamps ({n_timestamps}) for sequence length {self.sequence_length}")
        
        # Keep only valid timestamps
        if isinstance(data.index, pd.MultiIndex):
            data = data.loc[valid_timestamps]
        else:
            # For single index, keep blocks of n_assets rows for valid timestamps
            valid_indices = []
            for ts in valid_timestamps:
                start_idx = list(unique_timestamps).index(ts) * n_assets
                valid_indices.extend(range(start_idx, start_idx + n_assets))
            data = data.iloc[valid_indices]
        
        # Preprocess features
        data = self.preprocess_features(data)
        
        # Basic features
        features = ['return', 'volatility', 'momentum', 'price', 'price_to_sma',
                   'price_std', 'rsi', 'macd', 'macd_signal', 'bb_position']
        
        # Add all complex features
        features.extend(['skewness', 'kurtosis'])
        
        # Additional features
        for horizon in [1, 3, 5, 10]:
            data[f'price_change_{horizon}'] = (
                data['price'].pct_change(horizon).fillna(0)
            )
            features.append(f'price_change_{horizon}')
        
        # Volatility ratios and technical features
        data['vol_ratio_5_20'] = (
            data['return'].rolling(5).std() / 
            data['return'].rolling(20).std()
        ).fillna(1)
        features.append('vol_ratio_5_20')
        
        data['rsi_change'] = data['rsi'].diff().fillna(0)
        features.append('rsi_change')
        
        data['macd_hist'] = data['macd'] - data['macd_signal']
        features.append('macd_hist')
        
        # Cross-sectional rank features
        for feat in ['return', 'volatility', 'momentum', 'rsi']:
            data[f'{feat}_rank'] = data.groupby(level=0)[feat].rank(pct=True)
            features.append(f'{feat}_rank')
        
        # Check if all features exist in data
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(data[features])
        n_features = len(features)
        
        # Verify data structure
        if isinstance(data.index, pd.MultiIndex):
            group_sizes = data.groupby(level=0).size()
        else:
            # For single index, verify we have complete blocks of n_assets
            total_rows = len(data)
            if total_rows % n_assets != 0:
                raise ValueError(f"Data length ({total_rows}) is not divisible by number of assets ({n_assets})")
            group_sizes = pd.Series([n_assets] * (total_rows // n_assets))
        
        if group_sizes.min() != group_sizes.max() or group_sizes.min() != n_assets:
            raise ValueError("Inconsistent number of assets per timestamp after filtering!")
        
        # Reshape features into 3D array: [timestamps, assets, features]
        features_3d = features_scaled.reshape(n_timestamps, n_assets, n_features)
        
        # Get returns and reshape
        returns = data['return'].values.reshape(n_timestamps, n_assets)
        
        # Calculate future returns (shift up by 1 timestamp)
        future_returns = np.roll(returns, -1, axis=0)
        # Fill last row with the second-to-last row to handle the roll
        future_returns[-1] = future_returns[-2]
        
        # Create sequences and targets
        X, y = [], []
        
        # Adjust the range to account for sequence length and ensure we don't go out of bounds
        valid_indices = n_timestamps - self.sequence_length - 1
        
        if valid_indices <= 0:
            raise ValueError(f"No valid indices for sequence creation. Need more data points.")
        
        for i in range(valid_indices):
            try:
                # Input sequence
                features_batch = features_3d[i:i + self.sequence_length]
                X.append(features_batch)
                
                # Target - use future returns for next timestamp
                future_rets = future_returns[i + self.sequence_length]
                
                # Calculate optimal weights based on future returns
                weights = np.maximum(future_rets - np.mean(future_rets), 0)
                total_weight = np.sum(weights)
                if total_weight > 0:
                    weights = weights / total_weight
                else:
                    weights = np.ones(n_assets) / n_assets
                
                # Ensure weights match the number of assets
                if len(weights) != n_assets:
                    continue
                
                y.append(weights)
                
            except Exception:
                continue
        
        if not X:
            raise ValueError("No sequences created. Check sequence length and data availability.")
            
        X = np.array(X)
        y = np.array(y)
        
        # Add minimal noise in test mode
        if self.training and TEST_MODE:
            noise = np.random.normal(0, 0.001, X.shape)
            X = X + noise
        elif self.training:
            noise = np.random.normal(0, 0.01, X.shape)
            X = X + noise
        
        return X, y

    def predict_weights(self, portfolio: Portfolio, 
                       current_prices: Dict[str, dict]) -> np.ndarray:
        """Predict optimal weights using LSTM"""
        if self.model is None:
            return None

        # Extract features
        features = self.extract_features(portfolio, current_prices)
        if len(features) < self.sequence_length:
            return None

        # Remove asset_index before scaling
        if 'asset_index' in features.columns:
            asset_indices = features.pop('asset_index')
        
        # Scale features
        scaled_features = self.feature_scaler.transform(features)
        
        # Get dimensions
        n_assets = len(portfolio.portfolio_df)
        n_features = scaled_features.shape[1]
        total_features = n_assets * n_features
        
        # First reshape to [sequence_length, n_assets, n_features]
        try:
            X = scaled_features.reshape(-1, n_assets, n_features)
            if len(X) > self.sequence_length:
                # Take the last sequence_length rows
                X = X[-self.sequence_length:]
            elif len(X) < self.sequence_length:
                print(f"Warning: Not enough data points ({len(X)}) for sequence length ({self.sequence_length})")
                return None
            
            # Then reshape to [1, sequence_length, total_features]
            X = X.reshape(1, self.sequence_length, total_features)
            
            # Make prediction
            predicted_weights = self.model.predict(X, verbose=0)
            
            # Ensure weights are positive and sum to 1
            predicted_weights = np.maximum(predicted_weights, 0)
            predicted_weights = predicted_weights / predicted_weights.sum()
            
            return predicted_weights[0]
            
        except (ValueError, IndexError) as e:
            print(f"Error in predict_weights: {str(e)}")
            print(f"Shapes - scaled_features: {scaled_features.shape}, n_assets: {n_assets}, n_features: {n_features}")
            return None

    def evaluate(self, data: pd.DataFrame) -> Dict:
        """Evaluate model performance on given dataset"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        # Remove asset_index if present
        if 'asset_index' in data.columns:
            data = data.drop('asset_index', axis=1)
            
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Get dimensions
        if len(X.shape) == 4:  # [samples, sequence_length, n_assets, n_features]
            n_samples, seq_len, n_assets, n_features = X.shape
            total_features = n_assets * n_features
            X = X.reshape(n_samples, seq_len, total_features)
        
        # Get the expected input shape from the model
        expected_features = self.model.input_shape[-1]
        
        if X.shape[-1] != expected_features:
            # Try to reshape if possible
            if X.shape[-1] * n_assets == expected_features:
                X = X.reshape(n_samples, seq_len, -1)
            else:
                raise ValueError(
                    f"Feature dimension mismatch. Model expects {expected_features} features "
                    f"but got {X.shape[-1]}. Original shape: {X.shape}, "
                    f"n_assets: {n_assets}, n_features: {n_features}"
                )
        
        # Ensure batch size is consistent
        batch_size = min(32, len(X))
        
        # Evaluate the model
        loss, mae = self.model.evaluate(X, y, batch_size=batch_size, verbose=0)
        
        return {
            'val_loss': loss,
            'val_mae': mae,
            'test_loss': loss,
            'test_mae': mae
        }

    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature preprocessing with outlier handling"""
        data = data.copy()
        
        # Handle missing values
        data = data.ffill().bfill()
        
        # Detect and handle outliers
        for col in data.select_dtypes(include=[np.number]).columns:
            mean = data[col].mean()
            std = data[col].std()
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std
            data[col] = data[col].clip(lower_bound, upper_bound)
        
        return data
