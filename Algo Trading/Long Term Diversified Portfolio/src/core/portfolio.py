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
            'equity': 0.65,
            'fixed_income': 0.30,
            'cash': 0.05
        }
        
        # Define sector weights within equity - Optimized based on analysis
        self.sector_weights = {
            'technology': 0.384,
            'healthcare': 0.031,
            'financials': 0.333,
            'consumer_discretionary': 0.031,
            'industrials': 0.031,
            'energy': 0.19
        }  # Total: 100%
        
        # Define scoring weights for each sector - Optimized for quality and risk-adjusted returns
        self.scoring_weights = {
            'technology': {
                'roic': 0.35,
                'profit_margin': 0.25,
                'market_cap': 0.15,
                'debt_to_equity': 0.15,
                'beta': 0.10
            },
            'healthcare': {
                'roic': 0.30,
                'profit_margin': 0.25,
                'market_cap': 0.15,
                'debt_to_equity': 0.15,
                'beta': 0.15
            },
            'financials': {
                'roic': 0.20,
                'profit_margin': 0.15,
                'market_cap': 0.15,
                'debt_to_equity': 0.15,
                'dividend_yield': 0.25,
                'beta': 0.10
            },
            'consumer_discretionary': {
                'roic': 0.30,
                'profit_margin': 0.20,
                'market_cap': 0.15,
                'debt_to_equity': 0.15,
                'beta': 0.20
            },
            'industrials': {
                'roic': 0.25,
                'profit_margin': 0.20,
                'market_cap': 0.15,
                'debt_to_equity': 0.20,
                'beta': 0.20
            },
            'energy': {
                'roic': 0.25,
                'profit_margin': 0.20,
                'market_cap': 0.15,
                'debt_to_equity': 0.20,
                'dividend_yield': 0.10,
                'beta': 0.10
            }
        }
        
        # Define stock universe - Updated with more mid-cap and quality small-cap stocks
        self.stock_universe = {
            'technology': [
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AVGO', 'ASML', 'CRM', 'ADBE',
                'CSCO', 'INTC', 'AMD', 'QCOM', 'TXN', 'MU', 'AMAT',
                'KLAC', 'LRCX', 'TER', 'CDNS', 'SNPS'
            ],
            'healthcare': [
                'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'LLY', 'DHR',
                'BMY', 'AMGN', 'GILD', 'ISRG', 'VRTX', 'REGN', 'CI',
                'HUM', 'ELV', 'CVS', 'DVA', 'ZTS'
            ],
            'financials': [
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP',
                'V', 'MA', 'C', 'USB', 'PNC', 'TFC', 'MMC',
                'COF', 'AXP', 'MET', 'PRU', 'AIG'
            ],
            'consumer_discretionary': [
                'AMZN', 'MCD', 'SBUX', 'NKE', 'HD', 'LOW', 'DIS', 'NFLX',
                'TSLA', 'MAR', 'BKNG', 'CMG', 'TGT', 'COST', 'TJX',
                'LVS', 'MGM', 'RCL', 'CCL', 'HLT'
            ],
            'industrials': [
                'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'UNP', 'UPS',
                'FDX', 'LMT', 'RTX', 'NOC', 'EMR', 'ETN', 'WM',
                'RSG', 'WCN', 'PCAR', 'CHRW', 'ODFL'
            ],
            'energy': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR',
                'MPC', 'VLO', 'PSX', 'OXY', 'DVN', 'EOG',
                'FANG', 'APA', 'NOV', 'FTI'
            ]
        }
        
        # Define recommended ETFs - Updated with more specific sector ETFs
        self.recommended_etfs = {
            'healthcare': 'XLV',  # Health Care Select Sector SPDR Fund
            'consumer_discretionary': 'XLY',  # Consumer Discretionary Select Sector SPDR Fund
            'industrials': 'XLI'  # Industrial Select Sector SPDR Fund
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
        Get target holdings for the portfolio.
        
        Note: Sector weights are calculated as a percentage of the equity portion (65% of portfolio).
        For example, a sector weight of 20% means 20% of the equity allocation, not 20% of total portfolio.
        """
        # Calculate asset class allocations
        equity_value = self.portfolio_value * self.asset_weights['equity']  # 65% of portfolio
        fixed_income_value = self.portfolio_value * self.asset_weights['fixed_income']  # 30% of portfolio
        cash_value = self.portfolio_value * self.asset_weights['cash']  # 5% of portfolio
        
        # Get equity holdings
        equity_holdings = {}
        total_commission = 0
        
        # Allocate by sector (weights are relative to equity portion)
        for sector, weight in self.sector_weights.items():
            if sector in ['technology', 'healthcare', 'financials', 'consumer_discretionary']:
                # Direct holdings
                sector_value = equity_value * weight  # weight is % of equity
                stocks = self.get_recommended_stocks(sector, sector_value)
                for stock in stocks:
                    equity_holdings[stock['symbol']] = stock
                    total_commission += stock.get('commission', 0)
            else:
                # ETF holdings
                etf_symbol = self.recommended_etfs.get(sector)
                if etf_symbol:
                    equity_holdings[etf_symbol] = {
                        'name': f"{sector.title()} ETF",
                        'weight': (equity_value * weight) / self.portfolio_value,  # Convert to % of total portfolio
                        'amount': equity_value * weight,
                        'sector': sector
                    }
        
        # Calculate total equity value including commissions
        total_equity_value = sum(holding['amount'] for holding in equity_holdings.values())
        
        # Normalize equity weights to account for commissions
        if total_equity_value > 0:
            for symbol in equity_holdings:
                equity_holdings[symbol]['weight'] = (equity_holdings[symbol]['amount'] / total_equity_value) * self.asset_weights['equity']
        
        # Add fixed income and cash
        holdings = {
            'equity': equity_holdings,
            'fixed_income': {
                'AGG': {
                    'name': 'iShares Core U.S. Aggregate Bond ETF',
                    'weight': self.asset_weights['fixed_income'] * 0.7,
                    'amount': fixed_income_value * 0.7
                },
                'TLT': {
                    'name': 'iShares 20+ Year Treasury Bond ETF',
                    'weight': self.asset_weights['fixed_income'] * 0.3,
                    'amount': fixed_income_value * 0.3
                }
            },
            'cash': {
                'USD': {
                    'name': 'Cash',
                    'weight': self.asset_weights['cash'],
                    'amount': cash_value
                }
            }
        }
        
        return holdings
    
    def calculate_portfolio_metrics(self, holdings: Dict) -> PortfolioMetrics:
        """
        Calculate portfolio-level metrics using historical data.
        
        Note: Sector weights are calculated as a percentage of the equity portion.
        The weights are normalized to sum to 100% of the equity allocation.
        """
        # Calculate weights
        equity_weight = sum(holding['weight'] for holding in holdings['equity'].values())
        fixed_income_weight = sum(holding['weight'] for holding in holdings['fixed_income'].values())
        cash_weight = holdings['cash']['USD']['weight']
        
        # Calculate sector weights (as percentage of equity only)
        sector_weights = {}
        total_equity_weight = 0.0
        
        # First calculate total equity weight
        for holding in holdings['equity'].values():
            total_equity_weight += holding['weight']
        
        # Calculate sector weights as percentage of equity
        for holding in holdings['equity'].values():
            sector = holding['sector']
            sector_weights[sector] = sector_weights.get(sector, 0) + holding['weight']
        
        # Normalize sector weights to sum to 100% of equity
        if total_equity_weight > 0:
            for sector in sector_weights:
                sector_weights[sector] = sector_weights[sector] / total_equity_weight
        
        # Get historical data for all holdings
        equity_symbols = list(holdings['equity'].keys())
        fixed_income_symbols = list(holdings['fixed_income'].keys())
        
        # Fetch historical data
        equity_data = self.get_portfolio_historical_data(equity_symbols)
        fixed_income_data = self.get_portfolio_historical_data(fixed_income_symbols)
        
        if not equity_data or not fixed_income_data:
            raise ValueError("Unable to fetch historical data for portfolio components")
        
        # Calculate returns for each asset
        returns_data = {}
        
        # Calculate equity returns
        for symbol, data in equity_data.items():
            if 'Close' not in data.columns:
                raise ValueError(f"Missing 'Close' prices for {symbol}")
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        # Calculate fixed income returns
        for symbol, data in fixed_income_data.items():
            if 'Close' not in data.columns:
                raise ValueError(f"Missing 'Close' prices for {symbol}")
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        if returns_df.empty:
            raise ValueError("No valid return data available for portfolio components")
        
        # Calculate portfolio weights
        weights = {}
        for symbol in equity_symbols:
            weights[symbol] = holdings['equity'][symbol]['weight']
        for symbol in fixed_income_symbols:
            weights[symbol] = holdings['fixed_income'][symbol]['weight']
        
        # Calculate portfolio metrics
        # Expected return (annualized)
        expected_return = returns_df.mean().dot(pd.Series(weights)) * 252
        
        # Expected volatility (annualized)
        expected_volatility = np.sqrt(
            returns_df.cov().dot(pd.Series(weights)).dot(pd.Series(weights))
        ) * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns_df).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min().min()
        
        return PortfolioMetrics(
            total_value=self.portfolio_value,
            equity_weight=equity_weight,
            fixed_income_weight=fixed_income_weight,
            cash_weight=cash_weight,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            sector_weights=sector_weights,
            style_weights={}  # Empty for now, will be implemented later
        )

    def calculate_sector_correlations(self, lookback_years: int = 5, end_date: str = None) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Calculate sector returns and correlation matrix.
        
        Args:
            lookback_years (int): Number of years of historical data to use
            end_date (str): End date for historical data in YYYY-MM-DD format. If None, uses current date.
            
        Returns:
            Tuple[Dict[str, float], pd.DataFrame]: Dictionary of sector Sharpe ratios and correlation matrix
        """
        if end_date:
            end_date = pd.Timestamp(end_date)
        else:
            end_date = datetime.now()
            
        start_date = end_date - timedelta(days=lookback_years*365)
        
        # Risk-free rate
        risk_free_rate = 0.02
        
        # Get all sector returns
        sector_returns = {}
        sector_sharpe_ratios = {}
        
        # Calculate returns for sectors with ETFs
        for sector, etf in self.recommended_etfs.items():
            try:
                ticker = yf.Ticker(etf)
                hist_data = ticker.history(start=start_date, end=end_date)
                if len(hist_data) > 0:
                    returns = hist_data['Close'].pct_change().dropna()
                    sector_returns[sector] = returns
                    
                    # Calculate Sharpe ratio
                    annual_return = returns.mean() * 252
                    annual_volatility = returns.std() * np.sqrt(252)
                    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
                    sector_sharpe_ratios[sector] = sharpe_ratio
            except Exception as e:
                print(f"Error calculating returns for {sector} ETF: {e}")
        
        # Calculate returns for sectors with individual stocks
        direct_sectors = ['technology', 'healthcare', 'financials', 'consumer_discretionary']
        for sector in direct_sectors:
            try:
                stocks = self.stock_universe[sector][:5]
                stock_returns = []
                for symbol in stocks:
                    ticker = yf.Ticker(symbol)
                    hist_data = ticker.history(start=start_date, end=end_date)
                    if len(hist_data) > 0:
                        returns = hist_data['Close'].pct_change().dropna()
                        stock_returns.append(returns)
                
                if stock_returns:
                    # Calculate equal-weighted portfolio returns
                    portfolio_returns = pd.concat(stock_returns, axis=1).mean(axis=1)
                    sector_returns[sector] = portfolio_returns
                    
                    # Calculate Sharpe ratio
                    annual_return = portfolio_returns.mean() * 252
                    annual_volatility = portfolio_returns.std() * np.sqrt(252)
                    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
                    sector_sharpe_ratios[sector] = sharpe_ratio
            except Exception as e:
                print(f"Error calculating returns for {sector} stocks: {e}")
        
        # Create correlation matrix
        returns_df = pd.DataFrame(sector_returns)
        correlation_matrix = returns_df.corr()
        
        return sector_sharpe_ratios, correlation_matrix

    def optimize_portfolio(self, lookback_years: int = 5, end_date: str = None) -> Dict[str, float]:
        # Reset all sector weights to zero
        self.sector_weights = {sector: 0.0 for sector in self.sector_weights}
        
        # Get sector Sharpe ratios and correlation matrix
        sharpe_ratios, correlation_matrix = self.calculate_sector_correlations(lookback_years, end_date)
        
        # Log sector Sharpe ratios for analysis
        logger.info("\nSector Sharpe Ratios:")
        logger.info("--------------------")
        for sector, ratio in sorted(sharpe_ratios.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{sector}: {ratio:.3f}")
        
        # Log correlation matrix
        logger.info("\nSector Correlations:")
        logger.info("-------------------")
        logger.info(correlation_matrix.round(2))
        
        # Convert to numpy arrays for optimization
        sectors = list(sharpe_ratios.keys())
        ratio_values = np.array([sharpe_ratios[sector] for sector in sectors])
        corr_matrix = correlation_matrix.values
        
        # Define optimization constraints
        def objective(weights):
            portfolio_return = np.sum(weights * ratio_values)
            portfolio_vol = np.sqrt(weights.dot(corr_matrix).dot(weights))
            portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            return -portfolio_sharpe
        
        def constraint_sum(weights):
            return np.sum(weights) - 0.65
        
        def constraint_max(weights):
            return 0.25 - weights
        
        def constraint_min(weights):
            return weights - 0.02
        
        # Initial guess (equal weights summing to 65%)
        n_sectors = len(sectors)
        initial_weights = np.ones(n_sectors) * (0.65 / n_sectors)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': constraint_sum},
            {'type': 'ineq', 'fun': constraint_max},
            {'type': 'ineq', 'fun': constraint_min}
        ]
        
        # Define bounds (0.02 to 0.25 for each weight)
        bounds = [(0.02, 0.25) for _ in range(n_sectors)]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            # Create dictionary of optimized weights
            optimized_weights = dict(zip(sectors, result.x))
            
            # Log optimized weights
            logger.info("\nOptimized Weights (as % of total portfolio):")
            logger.info("----------------------------------------")
            for sector, weight in sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"{sector}: {weight:.1%}")
            
            # Log weights as % of equity portion
            logger.info("\nWeights (as % of equity portion):")
            logger.info("--------------------------------")
            for sector, weight in sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True):
                equity_weight = weight / 0.65
                logger.info(f"{sector}: {equity_weight:.1%}")
            
            # Update sector weights
            self.sector_weights = optimized_weights
            
            # Calculate and log portfolio metrics
            holdings = self.get_target_holdings()
            metrics = self.calculate_portfolio_metrics(holdings)
            
            logger.info("\nPortfolio Metrics:")
            logger.info(f"Expected Return: {metrics.expected_return:.1%}")
            logger.info(f"Expected Volatility: {metrics.expected_volatility:.1%}")
            logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            logger.info(f"Maximum Drawdown: {metrics.max_drawdown:.1%}")
            
            return optimized_weights
        else:
            logger.warning("Optimization failed. Using current weights.")
            return self.sector_weights

    def optimize_stock_selection(self, lookback_years: int = 5, start_date: str = '2023-01-01', end_date: str = '2024-01-01') -> Dict:
        # Calculate the start date for historical data
        hist_end_date = pd.Timestamp(start_date)
        hist_start_date = hist_end_date - pd.DateOffset(years=lookback_years)
        
        # Get historical data for all stocks and ETFs
        all_stocks = []
        for sector in ['technology', 'financials', 'energy']:
            all_stocks.extend(self.stock_universe[sector])
        
        # Add ETFs to the list
        all_stocks.extend(self.recommended_etfs.values())
        
        # Get historical data for optimization
        historical_data = self.get_portfolio_historical_data(
            all_stocks,
            start_date=hist_start_date.strftime('%Y-%m-%d'),
            end_date=hist_end_date.strftime('%Y-%m-%d')
        )
        
        # Calculate returns for each stock
        returns_data = {}
        for symbol, data in historical_data.items():
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        returns_df = pd.DataFrame(returns_data)
        
        # Get stock information and calculate scores
        stocks_info = {}
        for sector in ['technology', 'financials', 'energy']:
            sector_stocks = []
            for symbol in self.stock_universe[sector]:
                info = self.get_stock_info(symbol)
                if info:
                    info.score = self.calculate_stock_score(info, sector)
                    sector_stocks.append(info)
            
            # Sort by score and take top N
            sector_stocks.sort(key=lambda x: x.score, reverse=True)
            n_stocks = 5 if sector in ['technology', 'financials'] else 3
            stocks_info[sector] = sector_stocks[:n_stocks]
        
        # Prepare optimization data
        selected_symbols = []
        for sector, stocks in stocks_info.items():
            selected_symbols.extend([stock.symbol for stock in stocks])
        
        # Add ETF symbols
        for sector, etf in self.recommended_etfs.items():
            selected_symbols.append(etf)
        
        # Get returns for selected symbols
        selected_returns = returns_df[selected_symbols]
        
        # Calculate mean returns and covariance matrix
        mean_returns = selected_returns.mean() * 252  # Annualize
        cov_matrix = selected_returns.cov() * 252  # Annualize
        
        # Define optimization constraints
        def objective(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_vol = np.sqrt(weights.dot(cov_matrix).dot(weights))
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol  # Assuming 2% risk-free rate
            return -sharpe_ratio
        
        def constraint_sum(weights):
            return np.sum(weights) - 0.65  # Total equity weight
        
        def constraint_max(weights):
            return 0.15 - weights  # Maximum 15% per position
        
        def constraint_min(weights):
            return weights - 0.02  # Minimum 2% per position
        
        # Initial guess (equal weights)
        n_assets = len(selected_symbols)
        initial_weights = np.ones(n_assets) * (0.65 / n_assets)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': constraint_sum},
            {'type': 'ineq', 'fun': constraint_max},
            {'type': 'ineq', 'fun': constraint_min}
        ]
        
        # Define bounds (2% to 15% for each weight)
        bounds = [(0.02, 0.15) for _ in range(n_assets)]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            # Create holdings dictionary
            holdings = {
                'equity': {},
                'fixed_income': {
                    'AGG': {
                        'name': 'iShares Core U.S. Aggregate Bond ETF',
                        'weight': self.asset_weights['fixed_income'] * 0.7
                    },
                    'TLT': {
                        'name': 'iShares 20+ Year Treasury Bond ETF',
                        'weight': self.asset_weights['fixed_income'] * 0.3
                    }
                },
                'cash': {
                    'USD': {
                        'name': 'Cash',
                        'weight': self.asset_weights['cash']
                    }
                }
            }
            
            # Add equity holdings
            for i, symbol in enumerate(selected_symbols):
                weight = result.x[i]
                
                if symbol in self.recommended_etfs.values():
                    # ETF holding
                    sector = next(s for s, etf in self.recommended_etfs.items() if etf == symbol)
                    holdings['equity'][symbol] = {
                        'name': f"{sector.title()} ETF",
                        'weight': weight,
                        'sector': sector
                    }
                else:
                    # Stock holding
                    stock_info = next((s for s in sum(stocks_info.values(), []) if s.symbol == symbol), None)
                    if stock_info:
                        holdings['equity'][symbol] = {
                            'name': stock_info.name,
                            'weight': weight,
                            'sector': stock_info.sector,
                            'score': stock_info.score
                        }
            
            return holdings
        else:
            logger.warning("Optimization failed. Using current weights.")
            return self.get_target_holdings()

    def get_initial_positions(self, holdings: Dict, start_date: str = None, end_date: str = None) -> Dict:
        """
        Get initial positions with exact prices as of start_date.
        
        Args:
            holdings (Dict): Portfolio holdings from optimization
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
    
    # First optimize sector allocation using 5 years of data prior to 2022-12-30
    logger.info("\nOptimizing sector allocation using data from 2017-12-29 to 2022-12-30...")
    optimized_sector_weights = portfolio.optimize_portfolio(
        lookback_years=5,
        end_date='2022-12-30'
    )
    
    # Update portfolio's sector weights with the optimized weights
    if optimized_sector_weights:
        portfolio.sector_weights = optimized_sector_weights
        logger.info("\nUpdated sector weights after optimization:")
        for sector, weight in sorted(optimized_sector_weights.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{sector}: {weight:.1%}")
    else:
        logger.warning("Warning: Sector optimization did not return weights. Using default weights.")
    
    # Then optimize stock selection as of 2022-12-30
    logger.info("\nOptimizing stock selection as of 2022-12-30...")
    holdings = portfolio.optimize_stock_selection(
        lookback_years=5,
        start_date='2017-12-29',
        end_date='2022-12-30'
    )
    
    # Get initial positions with exact prices as of 2022-12-30
    logger.info("\nGetting initial positions as of 2022-12-30...")
    positions = portfolio.get_initial_positions(holdings, end_date='2022-12-30')
    
    # Log initial positions
    logger.info("\nInitial Positions:")
    logger.info("----------------")
    total_value = 0
    for symbol, pos in positions.items():
        total_value += pos['value']
    logger.info(f"\nTotal Portfolio Value: ${total_value:,.2f}")
    
    # Run backtest for 2023
    logger.info("\nRunning backtest for 2023...")
    backtest_results = portfolio.run_backtest(
        start_date='2022-12-30',
        end_date='2024-12-30',
        initial_capital=300_000
    )
    
    # Export position history to CSV
    portfolio.export_position_history(backtest_results, 'position_history_2023.csv')