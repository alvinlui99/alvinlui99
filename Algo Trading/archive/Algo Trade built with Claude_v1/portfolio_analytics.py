from typing import Dict, List
import pandas as pd
import numpy as np
from portfolio import Portfolio
from config import TradingConfig

class PortfolioAnalytics:
    """Handles performance metrics and portfolio analysis"""
    
    def __init__(self):
        # Data storage
        self.equity_curve: pd.DataFrame = pd.DataFrame()
        self.trades_df: pd.DataFrame = pd.DataFrame()
        
        # Performance metrics
        self.performance_metrics = {
            'total_return': 0.0,      # Total portfolio return in percentage
            'sharpe_ratio': 0.0,      # Risk-adjusted return metric
            'max_drawdown': 0.0,      # Maximum peak to trough decline
            'volatility': 0.0,        # Standard deviation of returns
            'win_rate': 0.0,          # Percentage of winning trades
            'profit_factor': 0.0,     # Ratio of gross profits to gross losses
            'total_trades': 0,        # Total number of trades
            'avg_trade': 0.0,         # Average P&L per trade
            'avg_win': 0.0,           # Average P&L of winning trades
            'avg_loss': 0.0           # Average P&L of losing trades
        }
        
        # Risk metrics
        self.risk_metrics = {
            'long_exposure': 0.0,     # Total long position value
            'short_exposure': 0.0,    # Total short position value
            'net_exposure': 0.0,      # Long - Short exposure
            'gross_exposure': 0.0,    # Long + Short exposure
            'cash_ratio': 1.0,        # Ratio of cash to total portfolio value
            'leverage': 0.0           # Gross exposure / Total value
        }

    def calculate_metrics(self, equity_curve: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics from equity curve and trades
        
        Args:
            equity_curve: DataFrame with columns ['timestamp', 'equity']
            trades_df: DataFrame containing trade records
            
        Returns:
            Dictionary containing performance metrics
        """
        # Store the data
        self.equity_curve = equity_curve
        self.trades_df = trades_df
        
        if self.equity_curve.empty:
            return self._empty_metrics()
            
        # Calculate returns
        returns = self.equity_curve['equity'].pct_change().fillna(0)
        
        # Calculate metrics
        metrics = {
            'total_return': self._calculate_total_return(),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(),
            'volatility': returns.std() * np.sqrt(TradingConfig.ANNUALIZATION_FACTOR)
        }
        
        # Add trade statistics if trades exist
        if not self.trades_df.empty:
            trade_stats = self._calculate_trade_stats()
            trade_stats['max_consecutive_losses'] = self._calculate_max_consecutive_losses()
            metrics.update(trade_stats)
            
        return metrics
    
    def analyze_risk(self, portfolio: Portfolio) -> Dict:
        """
        Calculate risk metrics for current portfolio state
        
        Args:
            portfolio: Portfolio instance
            
        Returns:
            Dictionary containing risk metrics
        """
        positions = {
            symbol: portfolio.get_position(symbol)
            for symbol in portfolio.assets.keys()
        }
        
        total_value = portfolio.get_total_value()
        if total_value == 0:
            return self._empty_risk_metrics()
        
        # Calculate exposure metrics
        long_exposure = sum(pos for pos in positions.values() if pos > 0)
        short_exposure = abs(sum(pos for pos in positions.values() if pos < 0))
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure
        
        return {
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'cash_ratio': portfolio.cash / total_value,
            'leverage': gross_exposure / total_value if total_value > 0 else 0
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary with all possible metrics initialized to zero"""
        return self.performance_metrics.copy()
    
    def _empty_risk_metrics(self) -> Dict:
        """Return empty risk metrics dictionary"""
        return self.risk_metrics.copy()
    
    def _calculate_total_return(self) -> float:
        """Calculate total return percentage"""
        if len(self.equity_curve) < 2:
            return 0.0
        return ((self.equity_curve['equity'].iloc[-1] / self.equity_curve['equity'].iloc[0]) - 1) * 100
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if returns.empty or returns.std() == 0:
            return 0.0
            
        excess_returns = returns - (TradingConfig.RISK_FREE_RATE / TradingConfig.ANNUALIZATION_FACTOR)
        annualized_returns = excess_returns.mean() * TradingConfig.ANNUALIZATION_FACTOR
        annualized_vol = returns.std() * np.sqrt(TradingConfig.ANNUALIZATION_FACTOR)
        
        return annualized_returns / annualized_vol if annualized_vol > 0 else 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if len(self.equity_curve) < 2:
            return 0.0
            
        equity = self.equity_curve['equity']
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak * 100
        
        return abs(drawdown.min())
    
    def _calculate_trade_stats(self) -> Dict:
        """Calculate trade-based statistics"""
        if self.trades_df.empty:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'avg_trade': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        # Calculate PnL for each trade
        self.trades_df['pnl'] = self.trades_df.apply(
            lambda x: (x['new_position'] - x['old_position']) * x['price'] - x['commission'],
            axis=1
        )
        
        winning_trades = self.trades_df[self.trades_df['pnl'] > 0]
        losing_trades = self.trades_df[self.trades_df['pnl'] < 0]
        
        total_trades = len(self.trades_df)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profits = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_losses = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        
        return {
            'win_rate': win_rate * 100,  # Convert to percentage
            'profit_factor': total_profits / total_losses if total_losses > 0 else float('inf'),
            'total_trades': total_trades,
            'avg_trade': self.trades_df['pnl'].mean(),
            'avg_win': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
            'avg_loss': losing_trades['pnl'].mean() if not losing_trades.empty else 0
        } 
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using only downside volatility"""
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (TradingConfig.RISK_FREE_RATE / TradingConfig.ANNUALIZATION_FACTOR)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(TradingConfig.ANNUALIZATION_FACTOR)
        
        return (excess_returns.mean() * TradingConfig.ANNUALIZATION_FACTOR) / downside_std if downside_std > 0 else 0.0
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades"""
        if self.trades_df.empty:
            return 0
        
        consecutive = 0
        max_consecutive = 0
        
        for pnl in self.trades_df['pnl']:
            if pnl < 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def clear_data(self):
        """Clear stored data and reset metrics"""
        self.equity_curve = pd.DataFrame()
        self.trades_df = pd.DataFrame() 