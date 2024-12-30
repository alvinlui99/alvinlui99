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
class PortfolioLoss(tf.keras.losses.Loss):
    """Custom portfolio loss function"""
    
    def __init__(self, name='portfolio_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
    
    def get_config(self):
        config = super().get_config()
        return config
    
    def call(self, y_true, y_pred):
        # Ensure predictions are in valid range
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Mean squared error
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Portfolio return
        returns = -tf.reduce_mean(y_pred * y_true)
        
        # Diversification penalty
        concentration = tf.reduce_sum(tf.square(y_pred), axis=1)
        diversification_penalty = 0.1 * tf.reduce_mean(concentration)
        
        # Combine losses
        total_loss = mse + 0.1 * returns + diversification_penalty
        
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
        
    def create_model(self, n_features: int):
        """Create simplified LSTM model with focus on stability"""
        model = tf.keras.Sequential([
            # Input layer with batch normalization
            tf.keras.layers.Input(shape=(self.sequence_length, n_features)),
            tf.keras.layers.BatchNormalization(),
            
            # First LSTM layer
            tf.keras.layers.LSTM(
                32,
                return_sequences=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Second LSTM layer
            tf.keras.layers.LSTM(
                16,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Dense layers
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Output layer
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        
        # Use a smaller learning rate
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,  # Reduced learning rate
            clipnorm=1.0,        # Gradient clipping
            epsilon=1e-7         # Numerical stability
        )
        
        model.compile(
            optimizer=optimizer,
            loss=PortfolioLoss(),
            metrics=[StableMAE()]  # Use our custom MAE metric
        )
        
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
        print("Starting model training...")
        self.training = True
        
        X, y = self.prepare_data(data)
        
        # Ensure targets are normalized
        y = y / np.sum(y, axis=1, keepdims=True)
        
        # Ensure targets are finite
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        if self.model is None:
            self.create_model(X.shape[2])
        
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
        
        print("Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=32,  # Fixed batch size for stability
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
        
        print("Loading best model...")
        self.model = tf.keras.models.load_model(
            'best_model.keras',
            custom_objects={
                'PortfolioLoss': PortfolioLoss,
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
        
        print("Training completed!")
        print(f"Final results: {results}")
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
        # Use smaller lookback in test mode
        if lookback is None:
            lookback = 20 if not TEST_MODE else 5
        
        features = []
        total_assets = len(portfolio.portfolio_df)
        
        for i, (symbol, row) in enumerate(portfolio.portfolio_df.iterrows()):
            try:
                asset = row['asset']
                returns = list(asset.get_return())
                prices = list(asset.prices)
                
                if len(returns) < lookback or len(prices) < lookback:
                    continue
                    
                current_price = current_prices.get(symbol, {}).get('markPrice')
                if current_price is None:
                    continue
                    
                # Basic features
                price_sma = np.mean(prices[-lookback:])
                price_std = np.std(prices[-lookback:])
                volatility = np.std(returns[-lookback:])
                momentum = sum(returns[-lookback:])
                
                # Only calculate complex features if not in test mode
                if not TEST_MODE:
                    skewness = pd.Series(returns[-lookback:]).skew()
                    kurtosis = pd.Series(returns[-lookback:]).kurtosis()
                    rsi = self.calculate_rsi(returns)
                    macd, macd_signal = self.calculate_macd(prices)
                    upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(prices)
                    bb_position = (current_price - lower_bb) / (upper_bb - lower_bb) if upper_bb != lower_bb else 0.5
                else:
                    # Simplified features for test mode
                    skewness = 0
                    kurtosis = 0
                    rsi = 50
                    macd, macd_signal = 0, 0
                    bb_position = 0.5
                
                price_to_sma = current_price / price_sma if price_sma != 0 else 1
                
                feature_dict = {
                    'return': returns[-1] if returns else 0,
                    'volatility': volatility,
                    'momentum': momentum,
                    'price': current_price,
                    'price_to_sma': price_to_sma,
                    'price_std': price_std,
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'bb_position': bb_position,
                    'skewness': skewness,
                    'kurtosis': kurtosis
                }
                features.append(feature_dict)
                
            except Exception as e:
                continue

        return pd.DataFrame(features)

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM with enhanced feature engineering"""
        print("Preparing data...")
        
        # Preprocess features first
        data = self.preprocess_features(data)
        
        # Basic features
        features = ['return', 'volatility', 'momentum', 'price', 'price_to_sma',
                   'price_std', 'rsi', 'macd', 'macd_signal', 'bb_position']
        
        # Only add complex features if not in test mode
        if not TEST_MODE:
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
        
        print(f"Number of features: {len(features)}")
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(data[features])
        target_scaled = self.scaler.fit_transform(data[['weight']])
        
        # Create sequences
        print("Creating sequences...")
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(features_scaled[i:(i + self.sequence_length)])
            y.append(target_scaled[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Add minimal noise in test mode
        if self.training and TEST_MODE:
            noise = np.random.normal(0, 0.001, X.shape)
            X = X + noise
        elif self.training:
            noise = np.random.normal(0, 0.01, X.shape)
            X = X + noise
        
        print(f"Final data shape: X={X.shape}, y={y.shape}")
        return X, y

    def predict_weights(self, portfolio: Portfolio, 
                       current_prices: Dict[str, dict]) -> np.ndarray:
        """Predict optimal weights using LSTM"""
        if self.model is None:
            return None

        features = self.extract_features(portfolio, current_prices)
        if len(features) < self.sequence_length:
            return None

        scaled_features = self.feature_scaler.transform(features)
        X = scaled_features.reshape(1, self.sequence_length, -1)

        predicted_weights = self.model.predict(X, verbose=0)
        predicted_weights = self.scaler.inverse_transform(predicted_weights)

        # Ensure weights are positive and sum to 1
        predicted_weights = np.maximum(predicted_weights, 0)
        predicted_weights = predicted_weights / predicted_weights.sum()

        return predicted_weights[0]

    def evaluate(self, data: pd.DataFrame) -> Dict:
        """Evaluate model performance on given dataset"""
        X, y = self.prepare_data(data)
        
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Evaluate the model
        loss, mae = self.model.evaluate(X, y, verbose=0)
        
        return {
            'val_loss': loss,
            'val_mae': mae,
            'test_loss': loss,  # Same as val_loss when used for test set
            'test_mae': mae     # Same as val_mae when used for test set
        }

    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature preprocessing with outlier handling"""
        data = data.copy()
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Detect and handle outliers
        for col in data.select_dtypes(include=[np.number]).columns:
            mean = data[col].mean()
            std = data[col].std()
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std
            data[col] = data[col].clip(lower_bound, upper_bound)
        
        return data
