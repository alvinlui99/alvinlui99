"""
Main entry point for the algorithmic trading system.
"""
import os
import sys
import logging
import argparse
from datetime import datetime
import time

# Third-party imports
from binance.um_futures import UMFutures
from dotenv import load_dotenv

# Local application/library specific imports
from config import (
    BaseConfig, BinanceConfig, ModelConfig, 
    TradingConfig, LoggingConfig, DataConfig, setup_logging
)
from model.LGBMmodel import LGBMmodel
from strategy.LGBMstrategy import LGBMstrategy
from core.trading_cycle import TradingCycle

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Algorithmic Trading System')
    
    parser.add_argument('--test', action='store_true', default=BinanceConfig.USE_TESTNET,
                       help=f'Run in test mode (no real orders) [default: {BinanceConfig.USE_TESTNET}]')
    parser.add_argument('--interval', type=int, default=TradingConfig.DEFAULT_INTERVAL_MINUTES,
                       help=f'Trading cycle interval in minutes [default: {TradingConfig.DEFAULT_INTERVAL_MINUTES}]')
    parser.add_argument('--timeframe', type=str, default=DataConfig.DEFAULT_TIMEFRAME,
                       help=f'Candlestick timeframe [default: {DataConfig.DEFAULT_TIMEFRAME}]')
    parser.add_argument('--lookback', type=int, default=BaseConfig.DEFAULT_LOOKBACK,
                       help=f'Number of lookback periods for analysis [default: {BaseConfig.DEFAULT_LOOKBACK}]')
    parser.add_argument('--cycles', type=int, default=None,
                       help='Number of cycles to run (None for infinite)')
    parser.add_argument('--symbols', nargs='+', default=BaseConfig.SYMBOLS,
                       help=f'Trading symbols [default: {", ".join(BaseConfig.SYMBOLS[:3])} and {len(BaseConfig.SYMBOLS)-3} more]')
    parser.add_argument('--max_allocation', type=float, default=TradingConfig.MAX_ALLOCATION,
                       help=f'Maximum allocation per position [default: {TradingConfig.MAX_ALLOCATION}]')
    parser.add_argument('--risk_per_trade', type=float, default=TradingConfig.RISK_PER_TRADE,
                       help=f'Risk per trade as fraction of capital [default: {TradingConfig.RISK_PER_TRADE}]')
    parser.add_argument('--debug', action='store_true', default=BaseConfig.DEBUG,
                       help=f'Enable debug mode [default: {BaseConfig.DEBUG}]')
    
    return parser.parse_args()

def initialize_system(args):
    """Initialize the trading system components."""
    # Set up logging
    logger = setup_logging()
    
    # Enable debug mode if requested
    if args.debug:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG)
    
    logger.info(f"Initializing trading system v{BaseConfig.VERSION}")
    
    # Check for required environment variables
    api_key = BinanceConfig.API_KEY
    api_secret = BinanceConfig.API_SECRET
    
    if not api_key or not api_secret:
        logger.error("API key and secret are required. Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
        sys.exit(1)
    
    # Initialize Binance client
    base_url = BinanceConfig.TESTNET_URL if args.test else BinanceConfig.PRODUCTION_URL
    client = UMFutures(
        key=api_key, 
        secret=api_secret, 
        base_url=base_url,
        timeout=BinanceConfig.REQUEST_TIMEOUT
    )
    
    # Test API connection
    try:
        server_time = client.time()
        logger.info(f"Connected to Binance {'Testnet' if args.test else 'Production'}")
        logger.info(f"Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
    except Exception as e:
        logger.error(f"Failed to connect to Binance: {str(e)}")
        sys.exit(1)
    
    # Load trading model
    try:
        # IMPORTANT: Make sure to check symbol compatibility with trained model
        available_models = []
        for symbol in args.symbols:
            model_path = os.path.join(BaseConfig.MODEL_DIR, f"{ModelConfig.MODEL_FILE_PREFIX}{symbol}{ModelConfig.MODEL_FILE_EXTENSION}")
            if os.path.exists(model_path):
                available_models.append(symbol)
            else:
                logger.warning(f"No model file found for {symbol} at {model_path}")
        
        if not available_models:
            logger.error(f"No model files found in {BaseConfig.MODEL_DIR}. Please train the model first.")
            sys.exit(1)
            
        # Use only symbols that have trained models
        logger.info(f"Using {len(available_models)} symbols with available models: {available_models}")
        
        model = LGBMmodel(available_models, feature_config=ModelConfig.FEATURE_CONFIG, logger=logger)
        model.load_model(BaseConfig.MODEL_DIR)
        logger.info(f"Model loaded successfully for {len(available_models)} symbols")
        
        # Update args.symbols to match available models
        args.symbols = available_models
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        sys.exit(1)
    
    # Initialize strategy
    strategy = LGBMstrategy(model)
    strategy.threshold = TradingConfig.SIGNAL_THRESHOLD
    logger.info(f"Strategy initialized with signal threshold: {strategy.threshold}")
    
    # Initialize trading cycle
    trading_cycle = TradingCycle(
        client=client,
        strategy=strategy,
        symbols=args.symbols,
        timeframe=args.timeframe,
        lookback_periods=args.lookback,
        test_mode=args.test,
        max_allocation=args.max_allocation,
        logger=logger
    )
    logger.info("Trading cycle initialized")
    
    return trading_cycle, logger

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize system
    trading_cycle, logger = initialize_system(args)
    
    logger.info(f"Starting trading system with {args.interval} minute intervals")
    logger.info(f"Trading {len(args.symbols)} symbols: {', '.join(args.symbols)}")
    logger.info(f"Test mode: {'ON' if args.test else 'OFF - REAL TRADING ENABLED'}")
    
    if not args.test:
        # Double-check before running in production mode
        confirm = input("WARNING: You are about to run in PRODUCTION mode with REAL funds. Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Aborting trading system startup")
            sys.exit(0)
    
    try:
        # Run a single cycle first to test
        logger.info("Running initial test cycle")
        results = trading_cycle.run()
        
        if results['status'] == 'ERROR':
            logger.error("Initial test cycle failed. Aborting.")
            sys.exit(1)
            
        logger.info("Initial test cycle completed successfully. Starting scheduled trading.")
        
        # Start scheduled trading
        trading_cycle.run_scheduled(
            interval_minutes=args.interval,
            max_cycles=args.cycles
        )
        
    except KeyboardInterrupt:
        logger.info("Trading system interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        # Cleanup
        logger.info("Cleaning up resources")
        
        if not args.test:
            # Ask for confirmation before cancelling orders in production
            confirm = input("Cancel all open orders? (y/n): ")
            if confirm.lower() == 'y':
                trading_cycle.cancel_all_open_orders()
                logger.info("All open orders cancelled")
        else:
            # Auto-cancel in test mode
            trading_cycle.cancel_all_open_orders()
            logger.info("All test orders cancelled")
        
        logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    # Sample usag
    # python main.py --test --interval 60 --timeframe 1h 
    main()