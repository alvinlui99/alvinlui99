"""
Portfolio management module - handles allocation, selection, and analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import os
from scipy.optimize import minimize
import logging
import copy  # Add this import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class StockMetrics:
    """Metrics for individual stocks."""
    symbol: str
    name: str
    sector: str
    market_cap: float
    profit_margin: float
    debt_to_equity: float
    roic: float
    dividend_yield: float
    beta: float
    current_price: float
    score: float = 0.0


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics."""
    total_value: float
    equity_weight: float
    fixed_income_weight: float
    cash_weight: float
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    sector_weights: Dict[str, float]
    style_weights: Dict[str, float]


@dataclass
class BacktestMetrics:
    """Metrics for backtesting results."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    sector_returns: Dict[str, float]
    monthly_returns: pd.Series
    daily_returns: pd.Series
    equity_curve: pd.Series
    position_history: List[Dict]


class PortfolioBacktest:
    def __init__(self, 
                 positions: Dict,
                 start_date: str,
                 end_date: str):
        """
        Initialize backtest with initial positions.
        
        Args:
            positions (Dict): Initial positions with shares and prices
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        self.positions = positions
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.price_data = None
        self.daily_returns = []
        self.equity_curve = []
        self.position_history = []
        
    def run_backtest(self) -> BacktestMetrics:
        """Run the backtest simulation."""
        # Get price data for backtest period
        symbols = list(self.positions.keys())
        self.price_data = self._get_price_data(symbols)
        
        # Initialize tracking variables
        initial_value = sum(pos['value'] for pos in self.positions.values())
        self.equity_curve = []
        self.daily_returns = []
        
        # Run simulation day by day
        for date in self.price_data.index:
            self._process_day(date, self.price_data.loc[date])
            
        # Calculate metrics
        return self._calculate_metrics()
    
    def _get_price_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get price data for backtest period."""
        historical_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.start_date, end=self.end_date)
                if not data.empty:
                    historical_data[symbol] = data['Close']
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return pd.DataFrame(historical_data)
    
    def _process_day(self, date: pd.Timestamp, prices: pd.Series):
        """Process a single day of the backtest."""
        # Update position values
        total_value = 0
        for symbol, position in self.positions.items():
            if symbol in prices:
                position['current_price'] = prices[symbol]
                position['current_value'] = position['shares'] * prices[symbol]
                total_value += position['current_value']
            else:
                # If no price data, keep previous value
                total_value += position['current_value']
        
        # Calculate daily return
        if self.equity_curve:
            daily_return = (total_value / self.equity_curve[-1]) - 1
        else:
            # For first day, calculate return from initial value
            initial_value = sum(pos['shares'] * pos['price'] for pos in self.positions.values())
            daily_return = (total_value / initial_value) - 1
        
        self.daily_returns.append(daily_return)
        self.equity_curve.append(total_value)
        
        # Record position history with deep copy
        self.position_history.append({
            'date': date,
            'total_value': total_value,
            'positions': copy.deepcopy(self.positions)
        })
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate backtest performance metrics."""
        # Convert lists to Series
        daily_returns = pd.Series(self.daily_returns, index=self.price_data.index)
        equity_curve = pd.Series(self.equity_curve, index=self.price_data.index)
        
        # Calculate basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility  # Assuming 2% risk-free rate
        
        # Calculate drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate sector returns
        sector_returns = {}
        for position in self.positions.values():
            if 'sector' in position:
                sector = position['sector']
                initial_value = position['shares'] * position['price']
                final_value = position['shares'] * position['current_price']
                sector_returns[sector] = sector_returns.get(sector, 0) + (final_value - initial_value)
        
        # Log backtest results
        logger.info("\nBacktest Results:")
        logger.info("----------------")
        logger.info(f"Total Return: {total_return:.1%}")
        logger.info(f"Annualized Return: {annualized_return:.1%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.1%}")
        
        logger.info("\nSector Returns:")
        logger.info("--------------")
        for sector, returns in sector_returns.items():
            logger.info(f"{sector}: ${returns:,.2f}")
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            sector_returns=sector_returns,
            monthly_returns=daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1),
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            position_history=self.position_history
        )


class Portfolio:
    """Manages portfolio allocation, selection, and analysis."""
    
    def __init__(self, initial_capital: float):
        self.portfolio_value = initial_capital
        
        # Commission parameters
        self.commission_per_share = 0.005
        self.min_commission = 1.0
        self.max_commission_pct = 0.01
        
        # Define asset class weights
        self.asset_weights = {
            'tech': 0.20,  # 20% AI & Tech Equities
            'financials': 0.20,  # 20% Financials
            'fixed_income': 0.20,  # 20% Fixed Income
            'renewable_energy': 0.10,  # 10% Renewable Energy
            'commodities': 0.10,  # 10% Commodities
            'alternatives': 0.20  # 20% Alternatives (REITs, Infra)
        }
        
        # Define scoring weights for each sector - Optimized for quality and risk-adjusted returns
        self.scoring_weights = {
            'tech': {
                'roic': 0.35,
                'profit_margin': 0.25,
                'market_cap': 0.15,
                'debt_to_equity': 0.15,
                'beta': 0.10
            },
            'financials': {
                'roic': 0.20,
                'profit_margin': 0.15,
                'market_cap': 0.15,
                'debt_to_equity': 0.15,
                'dividend_yield': 0.25,
                'beta': 0.10
            },
            'renewable_energy': {
                'roic': 0.25,
                'profit_margin': 0.20,
                'market_cap': 0.15,
                'debt_to_equity': 0.20,
                'beta': 0.20
            }
        }
        
        # Define stock universe - Updated based on IPS
        self.stock_universe = {
            'tech': [
                'NVDA', 'AMD', 'TSM', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'ORCL',
                'AAPL', 'META', 'CRM', 'ADBE', 'INTC', 'QCOM', 'AVGO'

            ],
            'financials': [
                'JPM', 'GS', 'MS', 'HSBC', 'BNPQY', 'DBSDY', 'AON', 'CB',
                'MA', 'V', 'PYPL'
            ],
            'renewable_energy': [
                'ENPH', 'SEDG', 'ICLN', 'TAN', 'QCLN'
            ]
        }
    
    def get_stock_info(self, symbol: str) -> Optional[StockMetrics]:
        """Get detailed information about a stock."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return StockMetrics(
                symbol=symbol,
                name=info.get('longName', symbol),
                sector=info.get('sector', 'Unknown'),
                market_cap=info.get('marketCap', 0),
                profit_margin=info.get('profitMargins', 0),
                debt_to_equity=info.get('debtToEquity', float('inf')),
                roic=info.get('returnOnInvestedCapital', 0),
                dividend_yield=info.get('dividendYield', 0),
                beta=info.get('beta', 1.0),
                current_price=info.get('currentPrice', 0)
            )
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, start_date: str = None, end_date: str = None, period: str = None, interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol using yfinance.
        
        Args:
            symbol (str): The symbol to get data for
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            period (str): Period of data to fetch (e.g., '5y' for 5 years). Only used if start_date and end_date are None
            interval (str): Interval of data (e.g., '1d' for daily)
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with historical data or None if request fails
        """
        try:
            ticker = yf.Ticker(symbol)
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def get_portfolio_historical_data(self, symbols: List[str], start_date: str = None, end_date: str = None, period: str = None) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.
        
        Args:
            symbols (List[str]): List of symbols to get data for
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            period (str): Period of data to fetch. Only used if start_date and end_date are None
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their historical data
        """
        historical_data = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, start_date, end_date, period)
            if data is not None:
                historical_data[symbol] = data
        
        return historical_data
    
    def calculate_stock_score(self, stock: StockMetrics, sector: str) -> float:
        """Calculate a composite score for a stock based on multiple factors."""
        weights = self.scoring_weights[sector]
        
        # Handle missing or invalid values with safe defaults
        roic = max(0, float(stock.roic) if stock.roic is not None else 0)
        profit_margin = max(0, float(stock.profit_margin) if stock.profit_margin is not None else 0)
        market_cap = max(0, float(stock.market_cap) if stock.market_cap is not None else 0)
        debt_to_equity = min(10, max(0, float(stock.debt_to_equity) if stock.debt_to_equity is not None else 0))
        dividend_yield = max(0, float(stock.dividend_yield) if stock.dividend_yield is not None else 0)
        beta = min(3, max(0, float(stock.beta) if stock.beta is not None else 1.0))
        
        # Calculate score components
        score = (
            weights['roic'] * roic +
            weights['profit_margin'] * profit_margin +
            weights['market_cap'] * (market_cap / 1e12) +  # Normalize market cap to trillions
            weights['debt_to_equity'] * (1 / (1 + debt_to_equity)) +  # Invert debt-to-equity
            weights.get('dividend_yield', 0) * dividend_yield +
            weights['beta'] * (1 / beta)  # Invert beta
        )
        
        return score
    
    def calculate_commission(self, shares: int, price_per_share: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            shares (int): Number of shares
            price_per_share (float): Price per share
            
        Returns:
            float: Commission amount
        """
        # Calculate base commission
        base_commission = shares * self.commission_per_share
        
        # Apply minimum commission
        commission = max(self.min_commission, base_commission)
        
        # Apply maximum commission (1% of trade value)
        trade_value = shares * price_per_share
        max_commission = trade_value * self.max_commission_pct
        commission = min(commission, max_commission)
        
        return commission
    
    def get_recommended_stocks(self, sector: str, allocation_amount: float, 
                             max_stocks: int = 5) -> List[Dict]:
        """Get recommended stocks for a sector with position sizes."""
        # Get stock information
        stocks_info = []
        for symbol in self.stock_universe[sector]:
            info = self.get_stock_info(symbol)
            if info:
                info.score = self.calculate_stock_score(info, sector)
                stocks_info.append(info)
        
        if not stocks_info:
            return []
        
        # Sort by score
        stocks_info.sort(key=lambda x: x.score, reverse=True)
        
        # Take top N stocks
        selected_stocks = stocks_info[:max_stocks]
        
        # Calculate position sizes (equal weight for now)
        position_size = allocation_amount / len(selected_stocks)
        
        # Convert to dictionary format
        holdings = []
        for stock in selected_stocks:
            shares = int(position_size / stock.current_price)
            commission = self.calculate_commission(shares, stock.current_price)
            actual_position_size = (shares * stock.current_price) + commission
            
            holdings.append({
                'symbol': stock.symbol,
                'name': stock.name,
                'weight': actual_position_size / self.portfolio_value,
                'amount': actual_position_size,
                'shares': shares,
                'commission': commission,
                'sector': sector,
                'score': stock.score
            })
        
        return holdings
    
    def get_target_holdings(self) -> Dict:
        """
        Get target allocation percentages for the portfolio based on IPS.
        
        Returns:
            Dict: Dictionary containing allocation percentages for each asset class and their sub-allocations
        """
        # Define the base allocation percentages
        allocations = {
            'equity': {
                'tech': self.asset_weights['tech'],  # 20%
                'financials': self.asset_weights['financials'],  # 20%
                'renewable_energy': self.asset_weights['renewable_energy']  # 10%
            },
            'fixed_income': {
                'us_bonds': self.asset_weights['fixed_income'] * 0.5,  # 50% of fixed income (10% of portfolio)
                'international_bonds': self.asset_weights['fixed_income'] * 0.25,  # 25% of fixed income (5% of portfolio)
                'tips': self.asset_weights['fixed_income'] * 0.25  # 25% of fixed income (5% of portfolio)
            },
            'alternatives': {
                'reits': self.asset_weights['alternatives'] * 0.5,  # 50% of alternatives (10% of portfolio)
                'infrastructure': self.asset_weights['alternatives'] * 0.5  # 50% of alternatives (10% of portfolio)
            },
            'commodities': {
                'broad_commodities': self.asset_weights['commodities']  # 10% of portfolio
            }
        }
        
        return allocations

    def get_initial_positions(self, holdings: Dict, start_date: str = None, end_date: str = None) -> Dict:
        """
        Get initial positions with exact prices as of start_date.
        
        Args:
            holdings (Dict): Portfolio holdings from IPS
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            Dict: Initial positions with shares and prices
        """
        if start_date is None:
            start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        if end_date is None:
            end_date = (pd.to_datetime(start_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        positions = {}
        
        # Process equity holdings
        for symbol, holding in holdings['equity'].items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    shares = int((self.portfolio_value * holding['weight']) / price)
                    positions[symbol] = {
                        'shares': shares,
                        'price': price,
                        'value': shares * price,
                        'sector': holding['sector'],
                        'name': holding['name']
                    }
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
        
        # Process fixed income holdings
        for symbol, holding in holdings['fixed_income'].items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    shares = int((self.portfolio_value * holding['weight']) / price)
                    positions[symbol] = {
                        'shares': shares,
                        'price': price,
                        'value': shares * price,
                        'sector': 'fixed_income',
                        'name': holding['name']
                    }
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
        
        # Process alternative holdings
        for symbol, holding in holdings['alternatives'].items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    shares = int((self.portfolio_value * holding['weight']) / price)
                    positions[symbol] = {
                        'shares': shares,
                        'price': price,
                        'value': shares * price,
                        'sector': 'alternatives',
                        'name': holding['name']
                    }
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
                
        # Process commodity holdings
        for symbol, holding in holdings['commodities'].items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    shares = int((self.portfolio_value * holding['weight']) / price)
                    positions[symbol] = {
                        'shares': shares,
                        'price': price,
                        'value': shares * price,
                        'sector': 'commodities',
                        'name': holding['name']
                    }
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
        
        return positions

    def run_backtest(self, 
                    start_date: str,
                    end_date: str,
                    initial_capital: float) -> BacktestMetrics:
        """
        Run a backtest of the portfolio strategy.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            initial_capital (float): Initial capital for the backtest
            
        Returns:
            BacktestMetrics: Backtest results and metrics
        """
        # Get optimized holdings
        holdings = self.get_target_holdings()
        
        # Get initial positions with exact prices
        positions = self.get_initial_positions(holdings, end_date=start_date)
        
        # Run backtest
        backtest = PortfolioBacktest(
            positions=positions,
            start_date=start_date,
            end_date=end_date
        )
        
        return backtest.run_backtest()

    def export_position_history(self, backtest_results: BacktestMetrics, filename: str = 'position_history.csv'):
        """
        Export position history to CSV file.
        
        Args:
            backtest_results (BacktestMetrics): Results from backtest
            filename (str): Name of the CSV file to create
        """
        # Create a list to store flattened position data
        position_data = []
        
        # Process each day's position history
        for day_data in backtest_results.position_history:
            date = day_data['date']
            total_value = day_data['total_value']
            
            # Add each position's data
            for symbol, position in day_data['positions'].items():
                position_data.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Name': position['name'],
                    'Shares': position['shares'],
                    'Price': position['current_price'],
                    'Value': position['current_value'],
                    'Sector': position.get('sector', ''),
                    'Total_Portfolio_Value': total_value
                })
        
        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(position_data)
        df.to_csv(filename, index=False)
        logger.info(f"Position history exported to {filename}")


if __name__ == "__main__":
    # Example usage
    portfolio = Portfolio(initial_capital=300_000)
    
    # Get target holdings based on IPS
    holdings = portfolio.get_target_holdings()
    
    # Get initial positions
    positions = portfolio.get_initial_positions(holdings, end_date='2024-01-01')
    
    # Log initial positions
    logger.info("\nInitial Positions:")
    logger.info("----------------")
    total_value = 0
    for symbol, pos in positions.items():
        total_value += pos['value']
    logger.info(f"\nTotal Portfolio Value: ${total_value:,.2f}")