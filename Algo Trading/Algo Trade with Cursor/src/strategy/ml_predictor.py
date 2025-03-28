import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self, feature_window: int = 24, prediction_window: int = 4):
        """
        Initialize the ML predictor.
        
        Args:
            feature_window (int): Number of time steps to use for features
            prediction_window (int): Number of time steps to predict ahead
        """
        self.feature_window = feature_window
        self.prediction_window = prediction_window
        self.scaler = MinMaxScaler()
        self.model = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features from price data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with additional technical indicators
        """
        # Add technical indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['BB_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['BB_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        
        # Add price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Add volume-based features
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df

    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM model.
        
        Args:
            data (np.ndarray): Scaled feature data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y sequences
        """
        X, y = [], []
        for i in range(len(data) - self.feature_window - self.prediction_window + 1):
            X.append(data[i:(i + self.feature_window)])
            y.append(data[i + self.feature_window:i + self.feature_window + self.prediction_window, 0])
        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model for price prediction.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data
            
        Returns:
            Sequential: Compiled LSTM model
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(self.prediction_window)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> None:
        """
        Train the ML model.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        # Create features
        df = self.create_features(df)
        
        # Scale features
        feature_columns = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'BB_high', 'BB_low',
                          'Returns', 'Log_Returns', 'Volatility', 'Volume_Ratio']
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Prepare sequences
        X, y = self.prepare_sequences(scaled_data)
        
        # Build and train model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            df (pd.DataFrame): DataFrame with recent OHLCV data
            
        Returns:
            np.ndarray: Predicted prices
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return None
            
        # Create features and scale data
        df = self.create_features(df)
        feature_columns = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'BB_high', 'BB_low',
                          'Returns', 'Log_Returns', 'Volatility', 'Volume_Ratio']
        scaled_data = self.scaler.transform(df[feature_columns])
        
        # Prepare last sequence
        last_sequence = scaled_data[-self.feature_window:]
        last_sequence = last_sequence.reshape((1, self.feature_window, scaled_data.shape[1]))
        
        # Make prediction
        prediction = self.model.predict(last_sequence)
        
        # Inverse transform prediction
        prediction = self.scaler.inverse_transform(
            np.hstack([prediction, np.zeros((prediction.shape[0], scaled_data.shape[1] - 1))])
        )[:, 0]
        
        return prediction 