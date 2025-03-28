import pandas as pd
import logging
from pathlib import Path
import sys
from typing import Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from model.LGBMmodel import LGBMmodel
from strategy.LGBMstrategy import LGBMstrategy
from backtest_engine import BacktestEngine
from config import TradingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_backtest_data() -> Dict[str, pd.DataFrame]:
    """
    Load backtesting data for all symbols.
    
    Returns:
        Dictionary of DataFrames containing historical data for each symbol
    """
    data_dir = project_root / 'data' / 'backtest'
    data = {}
    
    # Load data for each symbol
    for file_path in data_dir.glob('*_backtest.csv'):
        symbol = file_path.stem.replace('_backtest', '')
        df = pd.read_csv(file_path)
        df['Open_time'] = pd.to_datetime(df['Open_time'])
        df.set_index('Open_time', inplace=True)
        data[symbol] = df
        
        logger.info(f"Loaded backtest data for {symbol}")
    
    return data

def main():
    """Main function to run the backtest."""
    try:
        # Load backtest data
        logger.info("Loading backtest data...")
        data = load_backtest_data()
        
        # Load trained models
        logger.info("Loading trained models...")
        symbols = list(data.keys())
        model = LGBMmodel(symbols=symbols, logger=logger)
        
        # Load each model
        model_dir = project_root / 'model' / 'trained_models'
        model.load_model(str(model_dir))
        
        # Initialize strategy
        logger.info("Initializing trading strategy...")
        strategy = LGBMstrategy(
            model=model,
            threshold=TradingConfig.SIGNAL_THRESHOLD,
            logger=logger
        )
        
        # Initialize and run backtest
        logger.info("Starting backtest...")
        engine = BacktestEngine(strategy=strategy)
        results = engine.run(data)
        
        # Print results
        logger.info("\nBacktest Results:")
        logger.info("-" * 50)
        metrics = results['metrics']
        logger.info(f"Initial Capital: ${engine.initial_capital:,.2f}")
        logger.info(f"Final Equity: ${metrics['final_equity']:,.2f}")
        logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
        logger.info(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        logger.info(f"Number of Trades: {metrics['num_trades']}")
        
        # Save detailed results
        results_dir = project_root / 'data' / 'backtest_results'
        results_dir.mkdir(exist_ok=True)
        
        # Save trades
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(results_dir / 'trades.csv', index=False)
        
        # Save equity curve
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df.to_csv(results_dir / 'equity_curve.csv', index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(results_dir / 'metrics.csv', index=False)
        
        logger.info(f"\nDetailed results saved to {results_dir}")
        
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 