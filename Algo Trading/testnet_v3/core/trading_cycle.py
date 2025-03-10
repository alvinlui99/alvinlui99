"""
Trading cycle module that orchestrates the entire algorithmic trading process.
"""
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from core.data_fetcher import DataFetcher
from core.portfolio_manager import PortfolioManager
from core.executor import OrderExecutor

class TradingCycle:
    """
    Manages the entire trading cycle from data fetching to execution.
    """
    
    def __init__(self, client, strategy, symbols: List[str], 
                timeframe: str = '1h', 
                lookback_periods: int = 100,
                test_mode: bool = True,
                max_allocation: float = 0.25,
                logger=None):
        """
        Initialize the trading cycle.
        
        Args:
            client: Binance Futures client
            strategy: Trading strategy instance
            symbols: List of trading symbols
            timeframe: Timeframe for klines data
            lookback_periods: Number of historical periods to analyze
            test_mode: Whether to run in test mode (no real orders)
            max_allocation: Maximum allocation per symbol
            logger: Optional logger instance
        """
        self.client = client
        self.strategy = strategy
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback_periods = lookback_periods
        self.test_mode = test_mode
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.data_fetcher = DataFetcher(client, symbols, logger=self.logger)
        self.portfolio_manager = PortfolioManager(
            client, symbols, max_allocation=max_allocation, logger=self.logger
        )
        self.executor = OrderExecutor(
            client, symbols, test_mode=test_mode, logger=self.logger
        )
        
        # Track last run time
        self.last_run_time = None
        self.cycle_count = 0
        
    def run(self) -> Dict[str, Any]:
        """
        Execute one complete trading cycle.
        
        Returns:
            Dictionary with results of the trading cycle
        """
        start_time = datetime.now()
        self.cycle_count += 1
        self.logger.info(f"Starting trading cycle #{self.cycle_count} at {start_time}")
        
        results = {
            'cycle_id': self.cycle_count,
            'start_time': start_time,
            'status': 'STARTED'
        }
        
        try:
            # 1. Fetch market data
            klines = self.data_fetcher.fetch_latest_klines(
                timeframe=self.timeframe,
                lookback_periods=self.lookback_periods
            )
            results['data_fetched'] = True
            results['symbols_fetched'] = list(klines.keys())
            
            # Check if we have valid data
            valid_data = all(not df.empty for df in klines.values())
            if not valid_data:
                missing_symbols = [symbol for symbol, df in klines.items() if df.empty]
                self.logger.warning(f"Missing data for symbols: {missing_symbols}")
                results['status'] = 'INCOMPLETE_DATA'
                results['missing_symbols'] = missing_symbols
                return results
                
            # 2. Generate signals from strategy
            signals = self.strategy.get_signals(klines)
            results['signals_generated'] = len(signals)
            results['signals'] = signals
            
            # 3. Construct portfolio
            trade_decisions = self.portfolio_manager.construct_portfolio(signals)
            results['trade_decisions'] = len(trade_decisions)
            
            # 4. Execute trades
            if trade_decisions:
                execution_results = self.executor.execute_trades(trade_decisions)
                results['trades_executed'] = len(execution_results)
                results['execution_results'] = execution_results
            else:
                self.logger.info("No trades to execute in this cycle")
                results['trades_executed'] = 0
            
            results['status'] = 'COMPLETED'
            
        except Exception as e:
            error_msg = f"Error in trading cycle: {str(e)}"
            self.logger.error(error_msg)
            results['status'] = 'ERROR'
            results['error'] = error_msg
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.last_run_time = end_time
        
        results['end_time'] = end_time
        results['duration_seconds'] = duration
        
        self.logger.info(f"Trading cycle #{self.cycle_count} completed in {duration:.2f} seconds")
        return results
    
    def run_scheduled(self, interval_minutes: int = 60, max_cycles: Optional[int] = None):
        """
        Run trading cycles at scheduled intervals.
        
        Args:
            interval_minutes: Minutes between trading cycles
            max_cycles: Maximum number of cycles to run (None for infinite)
        """
        cycle_count = 0
        self.logger.info(f"Starting scheduled trading with {interval_minutes} minute intervals")
        
        try:
            while True:
                # Run one trading cycle
                self.run()
                cycle_count += 1
                
                # Check if we've reached max cycles
                if max_cycles and cycle_count >= max_cycles:
                    self.logger.info(f"Reached maximum of {max_cycles} cycles, stopping")
                    break
                
                # Calculate next run time
                next_run = datetime.now() + timedelta(minutes=interval_minutes)
                sleep_seconds = (next_run - datetime.now()).total_seconds()
                
                if sleep_seconds > 0:
                    self.logger.info(f"Sleeping for {sleep_seconds:.2f} seconds until next cycle")
                    time.sleep(sleep_seconds)
                    
        except KeyboardInterrupt:
            self.logger.info("Trading schedule interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in scheduled trading: {str(e)}")
        
        self.logger.info(f"Scheduled trading completed after {cycle_count} cycles")
    
    def cancel_all_open_orders(self):
        """
        Cancel all open orders for all tracked symbols.
        
        Returns:
            Results of cancellation operation
        """
        self.logger.info("Cancelling all open orders")
        return self.executor.cancel_all_orders()
