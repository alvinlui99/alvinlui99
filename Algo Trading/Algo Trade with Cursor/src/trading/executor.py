import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import logging
import time
import os

from binance.um_futures import UMFutures

from data import BinanceDataCollector, DataProcessor
from utils import PairIndicators
from strategy import PairStrategy

logger = logging.getLogger(__name__)

class PairExecutor:
    def __init__(
        self,
        symbol1: str = 'BTCUSDT',
        symbol2: str = 'ETHUSDT',
        interval: str = '1h',
        window: int = 20,
        leverage: int = 10
    ):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.interval = interval
        self.window = window
        self.leverage = leverage

        # Initialize components
        self.collector = BinanceDataCollector()
        self.processor = DataProcessor()
        self.strategy = PairStrategy()
        self.client = UMFutures(
            key=os.getenv('BINANCE_API_KEY'),
            secret=os.getenv('BINANCE_API_SECRET'),
            base_url='https://testnet.binancefuture.com'
        )

        # State tracking
        self.current_position = {
            'in_position': False,
            'position_type': 0,  # 1 for long spread, -1 for short spread
            'entry_zscore': 0.0,
            'entry_price1': 0.0,
            'entry_price2': 0.0,
            'units1': 0.0,
            'units2': 0.0
        }
        
        # Performance tracking
        self.trades = []
        
    def fetch_latest_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch and process the latest data for both symbols."""
        try:
            # Get data for both symbols
            df1 = self.collector.get_historical_klines(
                self.symbol1,
                interval=self.interval,
                end_str=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                days_back=21
            )
            df2 = self.collector.get_historical_klines(
                self.symbol2,
                interval=self.interval,
                end_str=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                days_back=21
            )
            
            # Process individual assets
            df1 = self.processor.process_single_asset(df1)
            df2 = self.processor.process_single_asset(df2)
            
            # Align dataframes
            df1_aligned, df2_aligned = PairIndicators.align_dataframes(df1, df2)
            
            return df1_aligned, df2_aligned
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def execute_trade(self, signal: int, price1: float, price2: float, hedge_ratio: float):
        """Execute a trade based on the signal."""
        try:
            if signal == 0:
                if self.current_position['in_position']:
                    # Close existing position
                    if self.current_position['position_type'] == 1:
                        self.client.new_order(
                            symbol=self.symbol1,
                            side='SELL',
                            quantity=abs(self.current_position['units1']),
                            type='MARKET',
                            positionSide='BOTH'
                        )
                        self.client.new_order(
                            symbol=self.symbol2,
                            side='BUY',
                            quantity=abs(self.current_position['units2']),
                            type='MARKET',
                            positionSide='BOTH'
                        )
                    else:
                        self.client.new_order(
                            symbol=self.symbol1,
                            side='BUY',
                            quantity=abs(self.current_position['units1']),
                            type='MARKET',
                            positionSide='BOTH'
                        )
                        self.client.new_order(
                            symbol=self.symbol2,
                            side='SELL',
                            quantity=abs(self.current_position['units2']),
                            type='MARKET',
                            positionSide='BOTH'
                        )
                    
                    self.current_position['in_position'] = False
                    self.current_position['position_type'] = 0
                    logger.info("Position closed")
                return
                
            if not self.current_position['in_position']:
                # Calculate position sizes
                unit_cost = price1 + price2 * hedge_ratio
                position_size = float(self.client.account()['availableBalance']) * 0.99
                
                if signal == 1:
                    units1 = position_size / unit_cost * self.leverage
                else:
                    units1 = -position_size / unit_cost * self.leverage
                units2 = -units1 * hedge_ratio
                
                units1 = round(units1, 3)
                units2 = round(units2, 3)

                # Execute the trade
                if units1 > 0:
                    self.client.new_order(
                        symbol=self.symbol1,
                        side='BUY',
                        quantity=abs(units1),
                        type='MARKET',
                        positionSide='BOTH'
                    )
                    self.client.new_order(
                        symbol=self.symbol2,
                        side='SELL',
                        quantity=abs(units2),
                        type='MARKET',
                        positionSide='BOTH'
                    )
                else:
                    self.client.new_order(
                        symbol=self.symbol1,
                        side='SELL',
                        quantity=abs(units1),
                        type='MARKET',
                        positionSide='BOTH'
                    )
                    self.client.new_order(
                        symbol=self.symbol2,
                        side='BUY',
                        quantity=abs(units2),
                        type='MARKET',
                        positionSide='BOTH'
                    )
                
                # Update position state
                self.current_position.update({
                    'in_position': True,
                    'position_type': signal,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'units1': units1,
                    'units2': units2
                })
                
                logger.info(f"Opened {signal} position with {units1:.4f} {self.symbol1} and {units2:.4f} {self.symbol2}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            raise
    
    def run(self):
        """Main execution loop for live trading."""
        logger.info(f"Starting pair trading execution for {self.symbol1}/{self.symbol2}")
        
        self.client.change_leverage(symbol=self.symbol1, leverage=self.leverage)
        self.client.change_leverage(symbol=self.symbol2, leverage=self.leverage)

        while True:
            try:
                # Fetch latest data
                df1, df2 = self.fetch_latest_data()

                # Get current prices
                current_price1 = df1['close'].iloc[-1]
                current_price2 = df2['close'].iloc[-1]
                
                # Generate signals
                signals = self.strategy.generate_signals(
                    df1,
                    df2,
                    self.current_position
                )
                
                print(signals)

                # Execute trade if signal changes
                if signals['signal'] != self.current_position['position_type']:
                    self.execute_trade(
                        signals['signal'],
                        current_price1,
                        current_price2,
                        signals['hedge_ratio']
                    )

                # Log current state
                print("--------------------------------")
                print(f"Correlation: {signals['correlation']:.4f}")
                print(f"Z-Score: {signals['zscore']:.4f}")
                print(f"zscore_threshold: {signals['dynamic_threshold']:.4f}")
                print(f"volatility ratio: {signals['vol_ratio']:.4f}")
                print(f"Signal: {signals['signal']}")
                
                # Wait for next interval
                time.sleep(60)  # Adjust based on interval
                
            except Exception as e:
                logger.error(f"Error in main execution loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
                continue


if __name__ == "__main__":
    executor = PairExecutor()
    executor.run()