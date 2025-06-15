import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
from datetime import datetime, timedelta

from copula_based_strategy import CopulaBasedStrategy
from collector import BinanceDataCollector
from config import Config

class Backtest:
    def __init__(self, 
                 initial_capital: float = 100000,
                 start_date: str = None,
                 end_date: str = None,
                 coins: List[str] = None):
        """
        Initialize the backtesting environment.
        
        Args:
            initial_capital: Starting capital for the backtest
            start_date: Start date for backtesting (format: 'YYYY-MM-DD')
            end_date: End date for backtesting (format: 'YYYY-MM-DD')
        """
        self.initial_capital = initial_capital
        self.wallet_balance = initial_capital     # wallet balance does not include unrealised pnl
        self.margin_balance = initial_capital     # margin balance includes unrealised pnl
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S') if start_date else None
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S') if end_date else None
        
        self.strategy = CopulaBasedStrategy()
        self.collector = BinanceDataCollector()
        self.config = Config()
        self.coins = self.config.coins if coins is None else coins
        
        self.prices: Dict[str, float] = {}
        self.pairs: List[Dict[str, Dict[str, float]]] = [] # List of pairs with their size and entry price

        self.pnls: List[float] = []  # PnL history
        self.commissions: List[float] = []  # Commission history
        self.equity_curve: List[float] = [initial_capital]  # Portfolio value over time

        # For analysis
        self.max_pairs: int = 0
        self.trades: List[Dict] = []

        # Counter
        self.winning_trades: int = 0
        self.winning_trades_pnl: float = 0
        self.losing_trades: int = 0
        self.losing_trades_pnl: float = 0
        
    def calculate_position_size(self, c1: str, c2: str, price_1: float, price_2: float, hedge_ratio: float) -> Tuple[float, float]:
        """Calculate position sizes for a pair based on the hedge ratio."""
        investable_budget = self.margin_balance * self.config.investable_budget_pc
        unit_1 = investable_budget / (price_1 + price_2 * hedge_ratio)
        unit_2 = unit_1 * hedge_ratio
        return round(unit_1, 3), round(unit_2, 3)
    
    def calculate_unrealised_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate unrealised PnL based on current positions and prices."""
        pnl = 0
        for pair in self.pairs:
            for symbol, position in pair.items():
                pnl += position['size'] * (prices[symbol] - position['entry_price'])
        return pnl
    
    def get_hedge_ratio(self, c1: str, c2: str, data: Dict[str, pd.DataFrame], window: int = None) -> float:
        window = self.config.window if window is None else window
        hedge_ratio = np.sum(data[c1]['close'] * data[c2]['close']) / np.sum(data[c2]['close'] * data[c2]['close'])
        if hedge_ratio < 0:
            print(f"WARNING: Hedge ratio is negative for {c1}-{c2}")
        return hedge_ratio

    def run(self) -> Dict:
        """
        Run the backtest simulation.
        
        Returns:
            Dict containing backtest results and performance metrics
        """
        current_date = self.start_date
        hedge_ratio_negative_count = 0
        while current_date <= self.end_date:
            # Get trading signals
            data = self.collector.get_multiple_symbols_data(self.coins, end_str = current_date.strftime('%Y-%m-%d %H:%M:%S'))
            signals = self.strategy.run(data, self.pairs)

            for symbol in self.coins:
                if not data[symbol].empty:
                    self.prices[symbol] = data[symbol]['close'].iloc[-1]

            for signal in signals.values():
                c1, c2 = signal['c1'], signal['c2']
                mi = signal['mi']

                current_pair_bool = False
                for pair in self.pairs:
                    if c1 in pair.keys() and c2 in pair.keys():
                        current_pair_bool = True
                        pnl = pair[c1]['size'] * (self.prices[c1] - pair[c1]['entry_price']) + pair[c2]['size'] * (self.prices[c2] - pair[c2]['entry_price'])
                        # if (pair[c1]['size'] < 0 and mi < self.config.long_exit_threshold) or \
                        #     (pair[c1]['size'] > 0 and mi > self.config.short_exit_threshold) or \
                        if pnl > self.margin_balance * self.config.take_profit_pc or \
                            pnl < -self.margin_balance * self.config.stop_loss_pc:
                            # Exit the position
                            commission = (abs(pair[c1]['size']) * self.prices[c1] + abs(pair[c2]['size']) * self.prices[c2]) * self.config.commission_pc
                            if pnl > 0:
                                self.winning_trades += 1
                                self.winning_trades_pnl += pnl
                            else:
                                self.losing_trades += 1
                                self.losing_trades_pnl += pnl
                            self.pnls.append(pnl)
                            self.commissions.append(commission)
                            self.wallet_balance += pnl - commission

                            self.trades.append({
                                'c1': c1,
                                'c2': c2,
                                'size_1': pair[c1]['size'],
                                'size_2': pair[c2]['size'],
                                'hedge_ratio': np.nan,
                                'price_1': self.prices[c1],
                                'price_2': self.prices[c2],
                                'commission': commission,
                                'trade_date': current_date,
                                'side': 'long' if pair[c1]['size'] < 0 else 'short',
                                'entry_or_exit': 'exit',
                                'pnl': pnl
                            })

                            self.pairs.remove(pair)

                
                if not current_pair_bool and len(self.pairs) < self.config.max_positions:
                    if mi > self.config.long_threshold:
                        hedge_ratio = self.get_hedge_ratio(c1, c2, data)
                        if hedge_ratio < 0:
                            hedge_ratio_negative_count += 1
                        else:
                            unit_1, unit_2 = self.calculate_position_size(c1, c2, self.prices[c1], self.prices[c2], hedge_ratio)
                            self.pairs.append({
                                c1: {'size': -unit_1, 'entry_price': self.prices[c1]},
                                c2: {'size': unit_2, 'entry_price': self.prices[c2]}
                            })
                            commission = (abs(unit_1) * self.prices[c1] + abs(unit_2) * self.prices[c2]) * self.config.commission_pc
                            self.commissions.append(commission)
                            self.wallet_balance -= commission

                            self.trades.append({
                                'c1': c1,
                                'c2': c2,
                                'size_1': -unit_1,
                                'size_2': unit_2,
                                'hedge_ratio': hedge_ratio,
                                'entry_price_1': self.prices[c1],
                                'entry_price_2': self.prices[c2],
                                'commission': commission,
                                'trade_date': current_date,
                                'side': 'long',
                                'entry_or_exit': 'entry',
                                'pnl': 0
                            })
                    elif mi < self.config.short_threshold:
                        hedge_ratio = self.get_hedge_ratio(c1, c2, data)
                        if hedge_ratio < 0:
                            hedge_ratio_negative_count += 1
                        else:                                                   
                            unit_1, unit_2 = self.calculate_position_size(c1, c2, self.prices[c1], self.prices[c2], hedge_ratio)
                            self.pairs.append({
                                c1: {'size': unit_1, 'entry_price': self.prices[c1]},
                                c2: {'size': -unit_2, 'entry_price': self.prices[c2]}
                            })
                            commission = (abs(unit_1) * self.prices[c1] + abs(unit_2) * self.prices[c2]) * self.config.commission_pc
                            self.commissions.append(commission)
                            self.wallet_balance -= commission

                            self.trades.append({
                                'c1': c1,
                                'c2': c2,
                                'size_1': unit_1,
                                'size_2': -unit_2,
                                'hedge_ratio': hedge_ratio,
                                'entry_price_1': self.prices[c1],
                                'entry_price_2': self.prices[c2],
                                'commission': commission,
                                'trade_date': current_date,
                                'side': 'short',
                                'entry_or_exit': 'entry',
                                'pnl': 0
                            })
            
            # Calculate unrealised PnL
            unrealised_pnl = self.calculate_unrealised_pnl(self.prices)
            self.margin_balance = self.wallet_balance + unrealised_pnl
            self.equity_curve.append(self.margin_balance)
            
            print(f"Current date: {current_date}")
            print(f"Margin balance: {self.margin_balance}")
            print(f"Number of pairs: {len(self.pairs)}")
            print(f"Winning trades: {self.winning_trades}")
            print(f"Losing trades: {self.losing_trades}")
            print(f"Winning trades PnL: {self.winning_trades_pnl}")
            print(f"Losing trades PnL: {self.losing_trades_pnl}")

            self.max_pairs = max(self.max_pairs, len(self.pairs))
            current_date += timedelta(hours=1)
        
        # Calculate performance metrics
        total_return = (self.margin_balance - self.initial_capital) / self.initial_capital
        sharpe_ratio = pd.Series(self.pnls).mean() / pd.Series(self.pnls).std()
        
        return {
            'total_return': total_return,
            'ending_margin_balance': self.margin_balance,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(self.pnls),
            'pnls': sum(self.pnls),
            'max_pairs': self.max_pairs,
            'negative_hedge_ratio_count': hedge_ratio_negative_count,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'pnls': self.pnls,
            'commissions': self.commissions,            
        }

if __name__ == '__main__':
    # Record the start time
    start_time = datetime.now()

    backtest = Backtest(
        initial_capital=100000,
        start_date='2023-09-01 00:00:00',
        end_date='2023-10-31 23:59:59'
    )
    results = backtest.run()
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Total PnL: {sum(results['pnls']):.2f}")
    print(f"Max Number of Pairs: {results['max_pairs']}")
    print(f"Negative Hedge Ratio Count: {results['negative_hedge_ratio_count']}")

    # Export the equity curve, trades, pnls, and commissions to csv
    pd.DataFrame(results['equity_curve']).to_csv('equity_curve.csv', index=False)
    pd.DataFrame(results['trades']).to_csv('trades.csv', index=False)

    # Calculate the total runtime
    end_time = datetime.now()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime}")