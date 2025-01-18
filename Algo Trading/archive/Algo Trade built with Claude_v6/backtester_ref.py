from typing import List, Dict, Callable
import pandas as pd
from dataclasses import dataclass
from portfolio import Portfolio
from strategy import Strategy
import numpy as np
from config import TradingConfig
@dataclass
class TradeStats:
    """Store trade statistics and performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

class Backtester:
    def __init__(self, symbols: List[str], strategy: Strategy,
                 initial_capital: float = 10000):
        self.portfolio = Portfolio(symbols, initial_capital)
        self.strategy = strategy
        self.current_prices = {}
        self.equity_curve = pd.Series(index=pd.Index([], name='timestamp'), name='equity')
        self.leverage_curve = pd.Series(index=pd.Index([], name='timestamp'), name='leverage')
        self.total_commission = 0.0  # Track total commission paid
        self.portfolio_df = pd.DataFrame(index=pd.Index([], name='timestamp'), columns=list(self.portfolio.assets.keys()))
        self.commission_rate = TradingConfig.COMMISSION_RATE

        self.is_leveraged = True if hasattr(self.strategy, 'leverage_components') else False
        if self.is_leveraged:
            self.leverage_components = pd.DataFrame(
                index=pd.Index([], name='timestamp'),
                columns=list(self.strategy.leverage_components.keys())
            )

    def run(self, test_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run backtest simulation
        
        Args:
            test_data: Dictionary mapping symbols to their feature DataFrames.
                        Each DataFrame must have a 'price' column and share the same index.
            
        Returns:
            Dictionary containing backtest results
            
        Example:
            test_data = {
                'BTC': pd.DataFrame({
                    'price': [...],
                    'volume': [...],
                    'other_features': [...]
                }, index=timestamps),
                'ETH': pd.DataFrame({
                    'price': [...],
                    'volume': [...],
                    'other_features': [...]
                }, index=timestamps)
            }
        """
        # Get shared index from any DataFrame (they're all the same after validation)
        timestamps = next(iter(test_data.values())).index
        
        # Reset tracking at start of run
        self.total_commission = 0.0
        
        for timestamp in timestamps:
            # Create price dictionary for current timestamp
            self._update_prices({
                symbol: df.loc[timestamp] 
                for symbol, df in test_data.items()
            })
            
            # Get strategy signals
            signals = self.strategy(
                self.portfolio, 
                self.current_prices,
                test_data,
                self.equity_curve
            )
            if signals:
                self._execute_trades(signals, self.current_prices)
                if self.is_leveraged:
                    self._track_leverage_component(timestamp)
            # Track equity and positions
            self._track_equity(timestamp)
            self._track_portfolio(timestamp)
            self._track_leverage(timestamp)

        return self.get_results()

    def _execute_trades(self, signals: Dict[str, dict[str, float]], prices: Dict[str, float]) -> List[Dict[str, float]]:
        """
        Execute trades based on strategy signals
        """
        for symbol, signal in signals.items():
            original_position = self.portfolio.assets[symbol].position
            new_position = signal['quantity']

            if original_position != new_position:
                if original_position != 0:
                    self.portfolio.close_position(symbol, prices[symbol])
                if new_position != 0:
                    self.portfolio.open_position(symbol, prices[symbol], new_position, signal['leverage'])
                trade_value = abs(new_position - original_position) * prices[symbol]
                commission = trade_value * self.commission_rate
                self.total_commission += commission

    def _update_prices(self, current_data: Dict[str, pd.Series]) -> None:
        """
        Update current prices from data row
        
        Args:
            current_data: Dictionary mapping symbols to their current data Series
        """
        self.current_prices = {}
        for symbol in self.portfolio.assets.keys():
            try:
                data = current_data[symbol]
                price = data['price']
                
                if pd.isna(price):
                    raise KeyError(f"Price for {symbol} is NaN")
                    
                # Store price directly
                price = float(price)
                self.current_prices[symbol] = price
                
                # Update portfolio with direct price value
                self.portfolio.update_portfolio_pnl({symbol: price})
                
            except KeyError as e:
                raise ValueError(f"Price data not found for symbol {symbol}") from e

    def _track_equity(self, timestamp: pd.Timestamp) -> None:
        """
        Track portfolio equity value over time
        
        Args:
            timestamp: Current timestamp
        """
        self.equity_curve[timestamp] = self.portfolio.get_total_value(self.current_prices)

    def _track_portfolio(self, timestamp: pd.Timestamp) -> None:
        """
        Track positions, prices and values for each symbol
        
        Args:
            timestamp: Current timestamp
        """
        total_value = self.portfolio.get_total_value(self.current_prices)
        self.portfolio_df.loc[timestamp, 'total_portfolio_value'] = total_value

        total_leveraged_value = 0

        for symbol in self.portfolio.assets.keys():
            total_leveraged_value += self.portfolio.get_composition(self.current_prices)[symbol]['value']

        for symbol in self.portfolio.assets.keys():
            value = self.portfolio.get_composition(self.current_prices)[symbol]['value']
            self.portfolio_df.loc[timestamp, symbol] = value
            if total_leveraged_value == 0:
                self.portfolio_df.loc[timestamp, f'{symbol}_pct'] = 0
            else:
                self.portfolio_df.loc[timestamp, f'{symbol}_pct'] = value / total_leveraged_value * 100

    def _track_leverage(self, timestamp: pd.Timestamp) -> None:
        """
        Track leverage over time
        """
        self.leverage_curve[timestamp] = self.portfolio.leverage

    def _track_leverage_component(self, timestamp: pd.Timestamp) -> None:
        """Track leverage components over time"""
        self.leverage_components.loc[timestamp] = self.strategy.leverage_components

    def save_portfolio_to_csv(self, filepath: str) -> None:
        """
        Save detailed portfolio history to CSV file
        """
        self.portfolio_df.to_csv(filepath, index=False, float_format='%.6f')

    def get_results(self) -> Dict:
        """Results with commission analysis"""        
        # Calculate commission impact
        total_commission = self.total_commission
        final_equity = self.equity_curve.iloc[-1]
        commission_impact = (total_commission / final_equity) * 100
        leverage_components = self.leverage_components if self.is_leveraged else None

        return {
            'equity_curve': self.equity_curve,
            'leverage_curve': self.leverage_curve,
            'leverage_components': leverage_components,
            'commission_analysis': {
                'total_commission': total_commission,
                'commission_impact_pct': commission_impact
            }
        }