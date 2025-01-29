import pandas as pd
from typing import List, Tuple, Dict, Set
from config import ModelConfig, DATA_PATH, FeatureConfig

FEATURE_NAMES: List[str] = [
    'return',
    'volatility',
    'momentum',
    'price_to_sma',
    'price_std',
    'rsi',
    'macd',
    'macd_signal',
    'bb_position',
    'skewness',
    'kurtosis'
]

def load_historical_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Load historical data for each symbol into separate DataFrames"""
    symbol_data = {}
    all_timestamps = set()
    first_symbol = True
    
    for symbol in symbols:
        # Read and process data in one step
        df = (pd.read_csv(f"{DATA_PATH}/{symbol}.csv")
              .assign(datetime=lambda x: pd.to_datetime(x['index']))
              .set_index('datetime')
              .loc[start_date:end_date])
        
        # Create price DataFrame efficiently
        symbol_data[symbol] = pd.DataFrame(
            {'price': df['Close'],
             'high': df['High'],
             'low': df['Low'],
             'open': df['Open'],
             'volume': df['Volume']},
            index=df.index
        )
        
        # Track common timestamps
        if first_symbol:
            all_timestamps = set(df.index)
            first_symbol = False
        else:
            all_timestamps = all_timestamps.intersection(set(df.index))
    
    # Filter each DataFrame to include only common timestamps
    common_timestamps = sorted(list(all_timestamps))
    for symbol in symbols:
        symbol_data[symbol] = symbol_data[symbol].loc[common_timestamps]
    
    return symbol_data

def split_data_by_date(data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Split data into train, validation, and test sets by date for each symbol"""
    train_data = {}
    val_data = {}
    test_data = {}
    
    # Use first symbol to get date ranges (assuming all symbols have same dates due to earlier processing)
    first_symbol = list(data.keys())[0]
    first_df = data[first_symbol]
    n = len(first_df)
    
    train_end = int(n * ModelConfig.TRAIN_SIZE)
    val_end = train_end + int(n * ModelConfig.VALIDATION_SIZE)
    
    # Split each symbol's data
    for symbol, df in data.items():
        train_data[symbol] = df.iloc[:train_end]
        val_data[symbol] = df.iloc[train_end:val_end]
        test_data[symbol] = df.iloc[val_end:]
    
    # Print summary using first symbol's dates
    print(f"\nData split summary:")
    print(f"Training period: {train_data[first_symbol].index[0]} to {train_data[first_symbol].index[-1]}")
    print(f"Validation period: {val_data[first_symbol].index[0]} to {val_data[first_symbol].index[-1]}")
    print(f"Testing period: {test_data[first_symbol].index[0]} to {test_data[first_symbol].index[-1]}")
    
    return train_data, val_data, test_data

def engineer_features(data: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """Engineer features from price data"""
    features = pd.DataFrame(index=data.index)
    
    # Price and returns
    features['return'] = data[price_column].pct_change()
    features['price'] = data[price_column]
    features['high'] = data['high']
    features['low'] = data['low']
    features['open'] = data['open']
    features['volume'] = data['volume']
    
    # Technical indicators
    # Volatility
    features['volatility'] = features['return'].rolling(
        window=FeatureConfig.LOOKBACK_PERIOD).std()
    
    # Momentum
    features['momentum'] = features['return'].rolling(
        window=FeatureConfig.LOOKBACK_PERIOD).mean()
    
    # Price relative to SMA
    sma = features['price'].rolling(window=FeatureConfig.LOOKBACK_PERIOD).mean()
    features['price_to_sma'] = features['price'] / sma
    features['price_std'] = features['price'].rolling(
        window=FeatureConfig.LOOKBACK_PERIOD).std()
    
    # RSI
    delta = features['return']
    gain = (delta.where(delta > 0, 0)).rolling(
        window=FeatureConfig.TECHNICAL_INDICATORS['RSI_PERIOD']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(
        window=FeatureConfig.TECHNICAL_INDICATORS['RSI_PERIOD']).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = features['price'].ewm(
        span=FeatureConfig.TECHNICAL_INDICATORS['MACD_FAST'], adjust=False).mean()
    exp2 = features['price'].ewm(
        span=FeatureConfig.TECHNICAL_INDICATORS['MACD_SLOW'], adjust=False).mean()
    features['macd'] = exp1 - exp2
    features['macd_signal'] = features['macd'].ewm(
        span=FeatureConfig.TECHNICAL_INDICATORS['MACD_SIGNAL'], adjust=False).mean()
    
    # Bollinger Bands
    bb_sma = features['price'].rolling(
        window=FeatureConfig.TECHNICAL_INDICATORS['BB_PERIOD']).mean()
    bb_std = features['price'].rolling(
        window=FeatureConfig.TECHNICAL_INDICATORS['BB_PERIOD']).std()
    bb_upper = bb_sma + (FeatureConfig.TECHNICAL_INDICATORS['BB_STD'] * bb_std)
    bb_lower = bb_sma - (FeatureConfig.TECHNICAL_INDICATORS['BB_STD'] * bb_std)
    features['bb_position'] = (features['price'] - bb_lower) / (bb_upper - bb_lower)
    
    # Statistical moments
    features['skewness'] = features['return'].rolling(
        window=FeatureConfig.LOOKBACK_PERIOD).skew()
    features['kurtosis'] = features['return'].rolling(
        window=FeatureConfig.LOOKBACK_PERIOD).kurt()
    
    # Forward fill only and drop initial NaN rows
    features = features.ffill().dropna()
    
    return features

def prepare_features_for_symbols(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Prepare features for each symbol separately"""
    feature_data = {}
    
    for symbol, symbol_data in data.items():
        # Engineer features for this symbol
        feature_data[symbol] = engineer_features(symbol_data, 'price')
    
    return feature_data