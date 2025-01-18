from typing import List, Dict, Callable
import pandas as pd
from dataclasses import dataclass
from portfolio import Portfolio
from trade_executor import TradeExecutor
from portfolio_analytics import PortfolioAnalytics
from strategy import Strategy

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
        self.executor = TradeExecutor()
        self.analytics = PortfolioAnalytics()
        self.current_prices = {}
        self.equity_curve = pd.DataFrame(columns=['timestamp', 'equity'])
        self.trades_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'price', 'size', 'value', 
            'commission', 'type', 'old_position', 'new_position'
        ])
        
        # Enhanced positions DataFrame with more columns
        self.positions_df = pd.DataFrame(columns=[
            'timestamp',
            *[f'{symbol}_position' for symbol in symbols],  # Position sizes
            *[f'{symbol}_price' for symbol in symbols],     # Current prices
            *[f'{symbol}_value' for symbol in symbols],     # Position values
            'total_portfolio_value',                        # Total portfolio value
            'cash',                                         # Available cash
            'cumulative_commission'                         # Running total of commissions
        ])
        self.total_commission = 0.0  # Track total commission paid
        
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
                self.portfolio.update_price(symbol, price)
                
            except KeyError as e:
                raise ValueError(f"Price data not found for symbol {symbol}") from e

    def _track_equity(self, timestamp: pd.Timestamp) -> None:
        """
        Track portfolio equity value over time
        
        Args:
            timestamp: Current timestamp
        """
        current_equity = self.portfolio.get_total_value(self.current_prices)
        
        # Add to equity curve
        new_row = pd.DataFrame([{
            'timestamp': timestamp,
            'equity': current_equity
        }])
        self.equity_curve = pd.concat([self.equity_curve, new_row], ignore_index=True)
        
        # Update strategy's tracking (if implemented)
        self.strategy.track_portfolio_state(
            self.portfolio, 
            current_equity,
            timestep=len(self.equity_curve) - 1
        )

    def _track_positions(self, timestamp: pd.Timestamp) -> None:
        """
        Track enhanced portfolio state including positions, prices, and values
        
        Args:
            timestamp: Current timestamp
        """
        data = {'timestamp': timestamp}
        
        # Track positions, prices and values for each symbol
        for symbol in self.portfolio.assets.keys():
            position = self.portfolio.get_position(symbol)
            price = self.current_prices[symbol]
            value = position * price
            
            data[f'{symbol}_position'] = position
            data[f'{symbol}_price'] = price
            data[f'{symbol}_value'] = value
        
        # Add portfolio-level metrics
        data['total_portfolio_value'] = self.portfolio.get_total_value(self.current_prices)
        data['cash'] = self.portfolio.cash
        data['cumulative_commission'] = self.total_commission
        
        new_row = pd.DataFrame([data])
        self.positions_df = pd.concat([self.positions_df, new_row], ignore_index=True)

    def save_positions_to_csv(self, filepath: str) -> None:
        """
        Save detailed portfolio history to CSV file
        
        Args:
            filepath: Path where CSV file should be saved
            
        The CSV will contain:
        - Timestamp
        - For each symbol:
            - Position size
            - Price
            - Position value (in quote currency)
            - % of portfolio
        - Portfolio summary:
            - Total portfolio value
            - Cash balance
            - Cash %
            - Cumulative commission
        """
        # Create a copy of positions_df to avoid modifying original
        output_df = self.positions_df.copy()
        
        # Calculate percentage allocations for each symbol
        for symbol in self.portfolio.assets.keys():
            output_df[f'{symbol}_pct'] = (
                output_df[f'{symbol}_value'] / output_df['total_portfolio_value'] * 100
            )
        
        # Calculate cash percentage
        output_df['cash_pct'] = (
            output_df['cash'] / output_df['total_portfolio_value'] * 100
        )
        
        # Reorder columns for better readability
        ordered_columns = ['timestamp']
        
        # Add symbol-specific columns in groups
        for symbol in self.portfolio.assets.keys():
            ordered_columns.extend([
                f'{symbol}_value',
                f'{symbol}_pct'
            ])
        
        # Add portfolio summary columns at the end
        ordered_columns.extend([
            'total_portfolio_value',
            'cash',
            'cash_pct',
            'cumulative_commission'
        ])
        
        # Reorder and save
        output_df = output_df[ordered_columns]
        output_df.to_csv(filepath, index=False, float_format='%.6f')

    def get_results(self) -> Dict:
        """Enhanced results with commission analysis"""
        equity_curve = self.equity_curve.copy()
        
        # Calculate commission impact
        total_commission = self.total_commission
        final_equity = equity_curve['equity'].iloc[-1]
        commission_impact = (total_commission / final_equity) * 100
        
        # Calculate turnover
        total_trade_value = self.trades_df['value'].abs().sum()
        avg_portfolio_value = equity_curve['equity'].mean()
        turnover_ratio = (total_trade_value / 2) / avg_portfolio_value  # Divide by 2 to avoid double counting
        
        return {
            'equity_curve': equity_curve,
            'trades': self.trades_df,
            'positions': self.positions_df,
            'metrics': self.analytics.calculate_metrics(
                equity_curve, 
                self.trades_df
            ),
            'risk_metrics': self.analytics.analyze_risk(self.portfolio),
            'commission_analysis': {
                'total_commission': total_commission,
                'commission_impact_pct': commission_impact,
                'turnover_ratio': turnover_ratio,
                'avg_trade_size': self.trades_df['value'].mean(),
                'trade_count': len(self.trades_df)
            }
        }

    def _validate_test_data(self, test_data: Dict[str, pd.DataFrame]) -> None:
        """
        Validate test data format and contents
        
        Args:
            test_data: Dictionary mapping symbols to their feature DataFrames
            
        Raises:
            ValueError: If data validation fails
        """
        if not isinstance(test_data, dict):
            raise ValueError("test_data must be a dictionary mapping symbols to DataFrames")
        
        if not test_data:
            raise ValueError("test_data dictionary cannot be empty")
        
        # Validate symbols match portfolio
        missing_symbols = set(self.portfolio.assets.keys()) - set(test_data.keys())
        if missing_symbols:
            raise ValueError(f"Missing data for symbols: {missing_symbols}")
        
        # Validate all DataFrames have same index
        indices = [df.index for df in test_data.values()]
        if not all(indices[0].equals(idx) for idx in indices[1:]):
            raise ValueError("All DataFrames must share the same index")
        
        # Validate required price column exists
        for symbol, df in test_data.items():
            if 'price' not in df.columns:
                raise ValueError(f"Missing 'price' column for symbol {symbol}")

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
        # Validate input data
        self._validate_test_data(test_data)
        
        # Reset tracking DataFrames
        self.equity_curve = pd.DataFrame(columns=['timestamp', 'equity'])
        self.trades_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'price', 'size', 'value', 
            'commission', 'type', 'old_position', 'new_position'
        ])
        
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
                test_data,  # Pass full test data for feature access
                len(self.equity_curve)  # Current timestep
            )
            
            # Execute trades
            if signals:
                executed_trades = self.executor.execute_trades(
                    self.portfolio, 
                    signals, 
                    self.current_prices,
                    timestamp
                )
                
                # Record the executed trades and update total commission
                if executed_trades:
                    trades_df = pd.DataFrame(executed_trades)
                    self.trades_df = pd.concat([self.trades_df, trades_df], ignore_index=True)
                    self.total_commission += trades_df['commission'].sum()
            
            # Track equity and positions
            self._track_equity(timestamp)
            self._track_positions(timestamp)
            
        return self.get_results()