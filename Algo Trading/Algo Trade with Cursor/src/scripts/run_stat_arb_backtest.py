import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from binance.um_futures import UMFutures
from dotenv import load_dotenv
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.strategy.statistical_arbitrage import StatisticalArbitrageStrategy
from src.strategy.backtest import Backtest
from src.data.market_data import MarketData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data_from_csv(symbols: list, timeframe: str, data_dir: str) -> dict:
    """
    Load historical data from CSV files.
    
    Args:
        symbols: List of trading symbols
        timeframe: Data timeframe
        data_dir: Directory containing CSV files
        
    Returns:
        dict: Dictionary of DataFrames for each symbol
    """
    data = {}
    for symbol in symbols:
        try:
            # Find the most recent CSV file for this symbol and timeframe
            csv_files = [f for f in os.listdir(data_dir) 
                        if f.startswith(f"{symbol}_{timeframe}_") and f.endswith('.csv')]
            if not csv_files:
                logger.warning(f"No CSV files found for {symbol} {timeframe}")
                continue
                
            # Sort by date and get the most recent
            csv_files.sort()
            latest_file = csv_files[-1]
            
            # Load data
            df = pd.read_csv(os.path.join(data_dir, latest_file))
            
            # Convert column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Print actual columns for debugging
            logger.info(f"Columns in {latest_file}: {df.columns.tolist()}")
            
            # Convert timestamp to datetime and set as index
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns in {latest_file}")
                logger.error(f"Required columns: {required_columns}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                continue
                
            # Convert numeric columns
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Drop rows with NaN values
            df.dropna(inplace=True)
            
            # Sort index
            df.sort_index(inplace=True)
            
            # Validate date range - only check if data is too old
            if df.index[0].year < 2020:
                logger.error(f"Data too old for {symbol}: {df.index[0]} to {df.index[-1]}")
                continue
                
            data[symbol] = df
            logger.info(f"Loaded {len(df)} rows for {symbol} from {latest_file}")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            logger.error(f"Exception details: {str(e.__class__.__name__)}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    if not data:
        logger.error("No data was successfully loaded. Please check the CSV files and their column names.")
        raise ValueError("No data was successfully loaded. Please check the CSV files and their column names.")
            
    return data

def split_data(data: dict) -> tuple:
    """
    Split data into train, test, and validation sets.
    
    Args:
        data: Dictionary of DataFrames for each symbol
        
    Returns:
        tuple: (train_data, test_data, val_data)
    """
    # Find common date range across all symbols
    start_dates = []
    end_dates = []
    
    for symbol, df in data.items():
        start_dates.append(df.index[0])
        end_dates.append(df.index[-1])
        
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    logger.info(f"Common date range: {common_start} to {common_end}")
    
    if common_start >= common_end:
        logger.error("No valid date range found")
        return None, None, None
        
    # Calculate split points
    total_days = (common_end - common_start).days
    train_days = int(total_days * 0.6)
    test_days = int(total_days * 0.2)
    
    train_end = common_start + timedelta(days=train_days)
    test_end = train_end + timedelta(days=test_days)
    
    logger.info(f"Training period: {common_start} to {train_end}")
    logger.info(f"Testing period: {train_end} to {test_end}")
    logger.info(f"Validation period: {test_end} to {common_end}")
    
    # Split data
    train_data = {}
    test_data = {}
    val_data = {}
    
    for symbol, df in data.items():
        train_data[symbol] = df[(df.index >= common_start) & (df.index < train_end)]
        test_data[symbol] = df[(df.index >= train_end) & (df.index < test_end)]
        val_data[symbol] = df[(df.index >= test_end) & (df.index <= common_end)]
        
        # Log split sizes
        logger.info(f"\n{symbol} split sizes:")
        logger.info(f"Training: {len(train_data[symbol])} rows")
        logger.info(f"Testing: {len(test_data[symbol])} rows")
        logger.info(f"Validation: {len(val_data[symbol])} rows")
        
    return train_data, test_data, val_data

def backtest_strategy(data: dict, 
                     initial_capital: float = 10000.0,
                     max_position_size: float = 0.1,
                     stop_loss: float = 0.02,
                     take_profit: float = 0.03) -> dict:
    """
    Backtest the statistical arbitrage strategy.
    
    Args:
        data: Dictionary of DataFrames for each symbol
        initial_capital: Initial capital for backtesting
        max_position_size: Maximum position size as fraction of capital
        stop_loss: Stop loss percentage
        take_profit: Take profit percentage
        
    Returns:
        dict: Backtest results
    """
    # Find common timestamps across all pairs
    common_timestamps = None
    for symbol, df in data.items():
        if common_timestamps is None:
            common_timestamps = set(df.index)
        else:
            common_timestamps = common_timestamps.intersection(set(df.index))
            
    if not common_timestamps:
        logger.error("No common timestamps found across all pairs")
        logger.error("Available date ranges:")
        for symbol, df in data.items():
            logger.error(f"{symbol}: {df.index[0]} to {df.index[-1]}")
        raise ValueError("No common timestamps found across all pairs")
        
    # Convert to sorted list
    common_timestamps = sorted(list(common_timestamps))
    logger.info(f"Found {len(common_timestamps)} common timestamps")
    logger.info(f"Date range: {common_timestamps[0]} to {common_timestamps[-1]}")
    
    # Initialize results
    results = {
        'capital': [initial_capital],
        'positions': {},
        'trades': [],
        'returns': [],
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0
    }
    
    # Initialize positions
    for symbol in data.keys():
        results['positions'][symbol] = 0.0
        
    # Backtest loop
    for i in range(1, len(common_timestamps)):
        current_time = common_timestamps[i]
        previous_time = common_timestamps[i-1]
        
        try:
            # Calculate returns for each pair
            returns = {}
            for symbol, df in data.items():
                if current_time in df.index and previous_time in df.index:
                    returns[symbol] = (df.loc[current_time, 'close'] / 
                                     df.loc[previous_time, 'close'] - 1)
                    
            if not returns:
                logger.warning(f"No returns calculated for {current_time}")
                continue
                
            # Calculate portfolio return
            portfolio_return = sum(returns.values()) / len(returns)
            results['returns'].append(portfolio_return)
            
            # Update capital
            current_capital = results['capital'][-1] * (1 + portfolio_return)
            results['capital'].append(current_capital)
            
            # Check for stop loss or take profit
            if portfolio_return <= -stop_loss or portfolio_return >= take_profit:
                # Close all positions
                for symbol in results['positions']:
                    if results['positions'][symbol] != 0:
                        results['trades'].append({
                            'timestamp': current_time,
                            'symbol': symbol,
                            'type': 'close',
                            'price': data[symbol].loc[current_time, 'close'],
                            'size': results['positions'][symbol]
                        })
                        results['positions'][symbol] = 0.0
                        
        except Exception as e:
            logger.error(f"Error in backtest loop at {current_time}: {str(e)}")
            continue
            
    # Calculate performance metrics
    if results['returns']:
        returns_series = pd.Series(results['returns'])
        results['sharpe_ratio'] = returns_series.mean() / returns_series.std() * np.sqrt(252)
        results['max_drawdown'] = (returns_series.cumsum().expanding().max() - 
                                 returns_series.cumsum()).max()
                                 
    return results

def main():
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'TRUMPUSDT',
        'LINKUSDT', 'NEARUSDT', 'WIFUSDT', 'AVAXUSDT', '1000SHIBUSDT',
        'DOGEUSDT', '1000PEPEUSDT', 'WLDUSDT'
    ]
    timeframe = '15m'
    data_dir = 'data'
    
    # Load data
    logger.info("Loading data...")
    data = load_data_from_csv(symbols, timeframe, data_dir)
    
    if not data:
        logger.error("No data loaded. Exiting.")
        return
        
    # Split data into train, test, and validation sets
    logger.info("Splitting data...")
    train_data, test_data, val_data = split_data(data)
    
    if not train_data or not test_data or not val_data:
        logger.error("Data splitting failed. Exiting.")
        return
        
    # Initialize results list
    results = []
    
    # Training period
    logger.info("\nRunning training period backtest...")
    train_results = backtest_strategy(train_data)
    train_results['period'] = 'train'
    results.append(train_results)
    
    # Testing period
    logger.info("\nRunning testing period backtest...")
    test_results = backtest_strategy(test_data)
    test_results['period'] = 'test'
    results.append(test_results)
    
    # Validation period
    logger.info("\nRunning validation period backtest...")
    val_results = backtest_strategy(val_data)
    val_results['period'] = 'val'
    results.append(val_results)
    
    # Display results
    logger.info("\nBacktest Results Summary:")
    logger.info("=" * 50)
    
    for result in results:
        logger.info(f"\n{result['period'].upper()} Period:")
        logger.info(f"Initial Capital: ${result['capital'][0]:,.2f}")
        logger.info(f"Final Capital: ${result['capital'][-1]:,.2f}")
        logger.info(f"Total Return: {result['returns'][-1] * 100:.2f}%")
        logger.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {result['max_drawdown'] * 100:.2f}%")
        logger.info(f"Number of Trades: {len(result['trades'])}")
        
        # Calculate win rate
        if result['trades']:
            winning_trades = sum(1 for trade in result['trades'] if trade['type'] == 'close' and trade['size'] > 0)
            win_rate = winning_trades / len(result['trades'])
            logger.info(f"Win Rate: {win_rate:.2%}")
            
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        for result in results:
            plt.plot(result['capital'], label=result['period'].upper())
            
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('backtest_results.png')
        plt.close()
        
        logger.info("\nResults plot saved as 'backtest_results.png'")
        
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")
        
if __name__ == "__main__":
    main() 