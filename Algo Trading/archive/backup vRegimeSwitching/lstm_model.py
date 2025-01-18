import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
from config import ModelConfig, MODEL_PATH

class LSTMModel:
    """LSTM model for predicting asset returns"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = ModelConfig.SEQUENCE_LENGTH
        self.feature_columns = None
        self.target_columns = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model architecture"""
        inputs = tf.keras.Input(shape=input_shape)
        x = LSTM(units=ModelConfig.LSTM_UNITS, 
                 return_sequences=True)(inputs)
        x = Dropout(ModelConfig.DROPOUT_RATE)(x)
        x = LSTM(units=ModelConfig.LSTM_UNITS // 2)(x)
        x = Dropout(ModelConfig.DROPOUT_RATE)(x)
        x = Dense(units=ModelConfig.DENSE_UNITS, activation='relu')(x)
        outputs = Dense(units=1)(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=ModelConfig.LEARNING_RATE)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input sequences and target values"""
        # Scale features
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, 0])  # First column is returns
            
        return np.array(X), np.array(y)
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None) -> Dict[str, float]:
        """Train the LSTM model"""
        X_train, y_train = self.prepare_sequences(train_data)
        
        if val_data is not None:
            X_val, y_val = self.prepare_sequences(val_data)
        else:
            # Split training data for validation if no validation set provided
            split_idx = int(len(X_train) * 0.8)
            X_val, y_val = X_train[split_idx:], y_train[split_idx:]
            X_train, y_train = X_train[:split_idx], y_train[:split_idx]
        
        # Build model if not already built
        if self.model is None:
            self.build_model(input_shape=(self.sequence_length, X_train.shape[2]))
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=ModelConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=ModelConfig.EPOCHS,
            batch_size=ModelConfig.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0
        )
        
        # Get final metrics
        metrics = {
            'train_loss': float(history.history['loss'][-1]),
            'train_mae': float(history.history['mae'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_mae': float(history.history['val_mae'][-1])
        }
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Prepare sequences
        X, _ = self.prepare_sequences(data)
        
        # Generate predictions
        predictions = self.model.predict(X)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(
            np.concatenate([predictions, np.zeros((len(predictions), data.shape[1]-1))], axis=1)
        )[:, 0]
        
        return predictions
    
    def save_model(self, path: str = MODEL_PATH) -> None:
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        self.model.save(path)
    
    def load_model(self, path: str = MODEL_PATH) -> None:
        """Load model from disk"""
        try:
            self.model = load_model(path)
        except Exception as e:
            raise
