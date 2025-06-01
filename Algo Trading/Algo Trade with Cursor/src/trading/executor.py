import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import logging
import time
import os
import json

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
        leverage: int = 10,
        stop_loss_pct: float = 0.02
    ):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.interval = interval
        self.leverage = leverage

        self.stop_loss_pct = stop_loss_pct

        # Initialize components
        self.collector = BinanceDataCollector()
        self.processor = DataProcessor()
        self.strategy = PairStrategy()
        self.client = UMFutures(
            key=os.getenv('BINANCE_API_KEY'),
            secret=os.getenv('BINANCE_API_SECRET'),
            base_url='https://testnet.binancefuture.com'
        )
        
        # Create position file path
        self.position_file = f"position_{symbol1}_{symbol2}.json"
        # Initialize position file if it doesn't exist
        if not os.path.exists(self.position_file):
            self._save_position({
                'in_position': False,
                'position_type': 0,  # 1 for long spread, -1 for short spread
                'entry_total_value': 0.0,
                'entry_price1': 0.0,
                'entry_price2': 0.0,
                'units1': 0.0,
                'units2': 0.0,
                'last_stop_loss_signal': 0
            })
            self.current_position = self._load_position()
        
        # Performance tracking
        self.trades = []

    def _load_position(self) -> Dict:
        """Load current position state from file."""
        try:
            with open(self.position_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading position file: {str(e)}")
            return {
                'in_position': False,
                'position_type': 0,
                'entry_total_value': 0.0,
                'entry_price1': 0.0,
                'entry_price2': 0.0,
                'units1': 0.0,
                'units2': 0.0,
                'last_stop_loss_signal': 0
            }

    def _save_position(self, position: Dict):
        """Save current position state to file."""
        try:
            with open(self.position_file, 'w') as f:
                json.dump(position, f)
        except Exception as e:
            logger.error(f"Error saving position file: {str(e)}")
            raise

    def fetch_latest_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch and process the latest data for both symbols."""
        try:
            # Get data for both symbols
            df1 = self.collector.get_historical_klines(
                self.symbol1,
                interval=self.interval,
                end_str=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                days_back=60
            )
            df2 = self.collector.get_historical_klines(
                self.symbol2,
                interval=self.interval,
                end_str=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                days_back=60
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
    
    def execute_trade(self, signal: int, price1: float, price2: float, hedge_ratio: float = None):
        """Execute a trade based on the signal."""
        try:
            self.current_position = self._load_position()
            
            if signal == 0:
                if self.current_position['in_position']:
                    # Create trade_history directory if it doesn't exist
                    os.makedirs('trade_history', exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                    # Close existing position
                    side = 'SELL' if self.current_position['units1'] > 0 else 'BUY'
                    self.client.new_order(
                        symbol=self.symbol1,
                        side=side,
                        quantity=abs(self.current_position['units1']),
                        type='MARKET',
                        positionSide='BOTH'
                    )
                    trade_info = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': self.symbol1,
                        'signal': signal,
                        'price': price1,
                        'units': self.current_position['units1'],
                        'position_type': self.current_position['position_type']
                    }
                    
                    # Generate filename with date and pair information
                    filename = f"trade_history/trade_{self.symbol1}_{timestamp}.txt"

                    # Write trade info to a .txt file
                    with open(filename, 'a') as file:
                        file.write(f"{trade_info}\n")
                        
                    side = 'SELL' if self.current_position['units2'] > 0 else 'BUY'
                    self.client.new_order(
                        symbol=self.symbol2,
                        side=side,
                        quantity=abs(self.current_position['units2']),
                        type='MARKET',
                        positionSide='BOTH'
                    )
                    trade_info = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': self.symbol2,
                        'signal': signal,
                        'price': price2,
                        'units': self.current_position['units2'],
                        'position_type': self.current_position['position_type']
                    }
                    
                    # Generate filename with date and pair information
                    filename = f"trade_history/trade_{self.symbol2}_{timestamp}.txt"
                    
                    # Write trade info to a .txt file
                    with open(filename, 'a') as file:
                        file.write(f"{trade_info}\n")

                    # Update position state
                    self.current_position.update({
                        'in_position': False,
                        'position_type': 0
                    })
                    self._save_position(self.current_position)
                    logger.info("Position closed")
                return
                
            if not self.current_position['in_position'] and self.current_position['last_stop_loss_signal'] != signal:
                # Create trade_history directory if it doesn't exist
                os.makedirs('trade_history', exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                # Calculate position sizes
                
                position_size = float(self.client.account()['availableBalance']) * 0.99
                cost1 = position_size / (1 + hedge_ratio)
                cost2 = position_size - cost1
                
                units1 = signal * cost1 / price1 * hedge_ratio / abs(hedge_ratio)
                units2 = -signal * cost2 / price2 * hedge_ratio / abs(hedge_ratio)

                units1 = round(units1, 3)
                units2 = round(units2, 3)

                # Execute the trade
                side = 'BUY' if units1 > 0 else 'SELL'
                self.client.new_order(
                    symbol=self.symbol1,
                    side=side,
                    quantity=abs(units1),
                    type='MARKET',
                    positionSide='BOTH'
                )
                trade_info = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': self.symbol1,
                    'signal': signal,
                    'price': price1,
                    'units': self.current_position['units1'],
                    'position_type': self.current_position['position_type']
                }
                
                # Generate filename with date and pair information
                filename = f"trade_history/trade_{self.symbol1}_{timestamp}.txt"

                # Write trade info to a .txt file
                with open(filename, 'a') as file:
                    file.write(f"{trade_info}\n")

                side = 'BUY' if units2 > 0 else 'SELL'
                self.client.new_order(
                    symbol=self.symbol2,
                    side=side,
                    quantity=abs(units2),
                    type='MARKET',
                    positionSide='BOTH'
                )
                trade_info = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': self.symbol2,
                    'signal': signal,
                    'price': price2,
                    'units': self.current_position['units2'],
                    'position_type': self.current_position['position_type']
                }
                
                # Generate filename with date and pair information
                filename = f"trade_history/trade_{self.symbol2}_{timestamp}.txt"

                # Write trade info to a .txt file
                with open(filename, 'a') as file:
                    file.write(f"{trade_info}\n")

                # Update position state
                self.current_position.update({
                    'in_position': True,
                    'position_type': signal,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'units1': units1,
                    'units2': units2,
                    'entry_total_value': position_size
                })
                self._save_position(self.current_position)
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            raise
    
    def check_stop_loss(self, current_price1: float, current_price2: float):
        """Check if stop loss has been hit."""
        if self.current_position['in_position']:
            unrealised_pnl = self.current_position['units1'] * (current_price1 - self.current_position['entry_price1']) + self.current_position['units2'] * (current_price2 - self.current_position['entry_price2'])
            if unrealised_pnl < -self.stop_loss_pct * self.leverage * self.current_position['entry_total_value']:
                self.current_position['last_stop_loss_signal'] = self.current_position['position_type']
                self.execute_trade(0, current_price1, current_price2)
                self._save_position(self.current_position)

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
                
                # Load current position state
                self.current_position = self._load_position()

                # Generate signals
                signals = self.strategy.generate_signals(
                    df1.iloc[-900:],
                    df2.iloc[-900:],
                    self.current_position
                )
                
                # print(signals)

                self.check_stop_loss(current_price1, current_price2)
                
                # Execute trade if signal changes
                if signals['signal'] != self.current_position['position_type']:
                    self.execute_trade(
                        signals['signal'],
                        current_price1,
                        current_price2,
                        signals['hedge_ratio']
                    )
                
                # Wait for next interval
                time.sleep(60)  # Adjust based on interval
                
            except Exception as e:
                logger.error(f"Error in main execution loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
                continue


if __name__ == "__main__":
    executor = PairExecutor()
    executor.run()