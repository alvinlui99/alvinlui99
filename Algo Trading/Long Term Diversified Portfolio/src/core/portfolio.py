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
import matplotlib.pyplot as plt
import seaborn as sns


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
            print(f"Error fetching info for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = '5y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol using yfinance.
        
        Args:
            symbol (str): The symbol to get data for
            period (str): Period of data to fetch (e.g., '5y' for 5 years)
            interval (str): Interval of data (e.g., '1d' for daily)
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with historical data or None if request fails
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            return df
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def get_portfolio_historical_data(self, symbols: List[str], period: str = '5y') -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.
        
        Args:
            symbols (List[str]): List of symbols to get data for
            period (str): Period of data to fetch
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their historical data
        """
        historical_data = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, period)
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

    def export_to_csv(self, holdings: Dict, metrics: PortfolioMetrics, output_dir: str = "output/reports") -> None:
        """
        Export portfolio composition and metrics to CSV files.
        
        Args:
            holdings (Dict): Portfolio holdings
            metrics (PortfolioMetrics): Portfolio metrics
            output_dir (str): Directory to save CSV files (default: output/reports)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare holdings data
        holdings_data = []
        
        # Add equity holdings
        for symbol, holding in holdings['equity'].items():
            holdings_data.append({
                'Symbol': symbol,
                'Name': holding['name'],
                'Asset Class': 'Equity',
                'Sector': holding['sector'],
                'Weight': holding['weight'],
                'Amount': holding['amount'],
                'Shares': holding.get('shares', 'N/A'),
                'Commission': holding.get('commission', 'N/A'),
                'Score': holding.get('score', 'N/A')
            })
        
        # Add fixed income holdings
        for symbol, holding in holdings['fixed_income'].items():
            holdings_data.append({
                'Symbol': symbol,
                'Name': holding['name'],
                'Asset Class': 'Fixed Income',
                'Sector': 'Fixed Income',
                'Weight': holding['weight'],
                'Amount': holding['amount'],
                'Shares': 'N/A',
                'Commission': 'N/A',
                'Score': 'N/A'
            })
        
        # Add cash
        holdings_data.append({
            'Symbol': 'USD',
            'Name': 'Cash',
            'Asset Class': 'Cash',
            'Sector': 'Cash',
            'Weight': holdings['cash']['USD']['weight'],
            'Amount': holdings['cash']['USD']['amount'],
            'Shares': 'N/A',
            'Commission': 'N/A',
            'Score': 'N/A'
        })
        
        # Convert to DataFrame and save
        holdings_df = pd.DataFrame(holdings_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        holdings_df.to_csv(f"{output_dir}/portfolio_holdings_{timestamp}.csv", index=False)
        
        # Prepare metrics data
        metrics_data = {
            'Metric': [
                'Total Value',
                'Equity Weight',
                'Fixed Income Weight',
                'Cash Weight',
                'Expected Return',
                'Expected Volatility',
                'Sharpe Ratio',
                'Max Drawdown'
            ],
            'Value': [
                f"${metrics.total_value:,.2f}",
                f"{metrics.equity_weight:.1%}",
                f"{metrics.fixed_income_weight:.1%}",
                f"{metrics.cash_weight:.1%}",
                f"{metrics.expected_return:.1%}",
                f"{metrics.expected_volatility:.1%}",
                f"{metrics.sharpe_ratio:.2f}",
                f"{metrics.max_drawdown:.1%}"
            ]
        }
        
        # Add sector weights
        for sector, weight in metrics.sector_weights.items():
            metrics_data['Metric'].append(f"{sector.title()} Weight")
            metrics_data['Value'].append(f"{weight:.1%}")
        
        # Add style weights
        for style, weight in metrics.style_weights.items():
            metrics_data['Metric'].append(f"{style.replace('_', ' ').title()} Weight")
            metrics_data['Value'].append(f"{weight:.1%}")
        
        # Convert to DataFrame and save
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f"{output_dir}/portfolio_metrics_{timestamp}.csv", index=False)
        
        # Calculate and export sector correlations
        equity_symbols = list(holdings['equity'].keys())
        equity_data = self.get_portfolio_historical_data(equity_symbols)
        
        if equity_data:
            # Calculate returns for each asset
            returns_data = {}
            for symbol, data in equity_data.items():
                if 'Close' in data.columns:
                    returns = data['Close'].pct_change().dropna()
                    returns_data[symbol] = returns
            
            returns_df = pd.DataFrame(returns_data)
            
            if not returns_df.empty:
                # Get unique sectors
                sectors = sorted(set(holdings['equity'][symbol]['sector'] for symbol in equity_symbols))
                
                # Create correlation matrix
                correlation_data = []
                for i, sector1 in enumerate(sectors):
                    for sector2 in sectors[i:]:  # Include same sector correlations
                        sector1_symbols = [symbol for symbol in equity_symbols 
                                         if holdings['equity'][symbol]['sector'] == sector1]
                        sector2_symbols = [symbol for symbol in equity_symbols 
                                         if holdings['equity'][symbol]['sector'] == sector2]
                        
                        if sector1_symbols and sector2_symbols:
                            # Calculate correlation between sectors
                            corr_matrix = returns_df[sector1_symbols + sector2_symbols].corr()
                            
                            # Get correlations between sectors
                            sector1_indices = range(len(sector1_symbols))
                            sector2_indices = range(len(sector1_symbols), len(sector1_symbols) + len(sector2_symbols))
                            
                            if sector1 == sector2:
                                # For same sector, use upper triangle excluding diagonal
                                corr_values = corr_matrix.iloc[sector1_indices, sector2_indices].values
                                corr_values = corr_values[np.triu_indices_from(corr_values, k=1)]
                            else:
                                # For different sectors, use all cross-correlations
                                corr_values = corr_matrix.iloc[sector1_indices, sector2_indices].values.flatten()
                            
                            correlation_data.append({
                                'Sector 1': sector1,
                                'Sector 2': sector2,
                                'Mean Correlation': np.mean(corr_values),
                                'Number of Pairs': len(corr_values)
                            })
                
                # Convert to DataFrame and save
                correlation_df = pd.DataFrame(correlation_data)
                correlation_df.to_csv(f"{output_dir}/sector_correlations_{timestamp}.csv", index=False)
                print(f"- Sector Correlations: sector_correlations_{timestamp}.csv")
        
        print(f"\nPortfolio data exported to {output_dir}/")
        print(f"- Holdings: portfolio_holdings_{timestamp}.csv")
        print(f"- Metrics: portfolio_metrics_{timestamp}.csv")

    def calculate_sector_correlations(self, lookback_years: int = 5) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Calculate sector returns and correlation matrix.
        
        Args:
            lookback_years (int): Number of years of historical data to use
            
        Returns:
            Tuple[Dict[str, float], pd.DataFrame]: Dictionary of sector Sharpe ratios and correlation matrix
        """
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

    def optimize_portfolio(self, lookback_years: int = 5) -> None:
        """
        Optimize portfolio weights using Sharpe ratio for the entire portfolio.
        Considers sector correlations in the optimization.
        
        Note: Sector weights are optimized as a percentage of the total portfolio.
        The weights must sum to 65% (equity portion) of the total portfolio.
        For example, if a sector has 20% weight, it means 20% of total portfolio,
        which is about 30.8% of the equity portion (20/65).
        
        Args:
            lookback_years (int): Number of years of historical data to use
        """
        # Reset all sector weights to zero
        self.sector_weights = {sector: 0.0 for sector in self.sector_weights}
        
        # Get sector Sharpe ratios and correlation matrix
        sharpe_ratios, correlation_matrix = self.calculate_sector_correlations(lookback_years)
        
        # Print sector Sharpe ratios for analysis
        print("\nSector Sharpe Ratios:")
        print("--------------------")
        for sector, ratio in sorted(sharpe_ratios.items(), key=lambda x: x[1], reverse=True):
            print(f"{sector}: {ratio:.3f}")
        
        # Print correlation matrix
        print("\nSector Correlations:")
        print("-------------------")
        print(correlation_matrix.round(2))
        
        # Convert to numpy arrays for optimization
        sectors = list(sharpe_ratios.keys())
        ratio_values = np.array([sharpe_ratios[sector] for sector in sectors])
        corr_matrix = correlation_matrix.values
        
        # Define optimization constraints
        def objective(weights):
            # Calculate portfolio return
            portfolio_return = np.sum(weights * ratio_values)
            
            # Calculate portfolio volatility using correlation matrix
            portfolio_vol = np.sqrt(weights.dot(corr_matrix).dot(weights))
            
            # Calculate portfolio Sharpe ratio
            portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            return -portfolio_sharpe  # Negative because we're minimizing
        
        def constraint_sum(weights):
            # Weights must sum to 0.65 (65% of total portfolio)
            return np.sum(weights) - 0.65
        
        def constraint_max(weights):
            # Maximum weight of 0.25 (25% of total portfolio)
            return 0.25 - weights
        
        def constraint_min(weights):
            # Minimum weight of 0.02 (2% of total portfolio)
            return weights - 0.02
        
        # Initial guess (equal weights summing to 65%)
        n_sectors = len(sectors)
        initial_weights = np.ones(n_sectors) * (0.65 / n_sectors)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': constraint_sum},  # Weights sum to 65%
            {'type': 'ineq', 'fun': constraint_max},  # Maximum weight
            {'type': 'ineq', 'fun': constraint_min}   # Minimum weight
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
            
            # Print optimized weights
            print("\nOptimized Weights (as % of total portfolio):")
            print("----------------------------------------")
            for sector, weight in sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True):
                print(f"{sector}: {weight:.1%}")
            
            # Print weights as % of equity portion
            print("\nWeights (as % of equity portion):")
            print("--------------------------------")
            for sector, weight in sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True):
                equity_weight = weight / 0.65  # Convert to % of equity portion
                print(f"{sector}: {equity_weight:.1%}")
            
            # Update sector weights
            self.sector_weights = optimized_weights
            
            # Calculate and print portfolio metrics
            holdings = self.get_target_holdings()
            metrics = self.calculate_portfolio_metrics(holdings)
            
            print("\nPortfolio Metrics:")
            print(f"Expected Return: {metrics.expected_return:.1%}")
            print(f"Expected Volatility: {metrics.expected_volatility:.1%}")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"Maximum Drawdown: {metrics.max_drawdown:.1%}")
            
            # Export results
            self.export_to_csv(holdings, metrics)
            
            return optimized_weights
        else:
            print("Optimization failed. Using current weights.")
            return self.sector_weights

    def calculate_forward_performance(self, holdings: Dict, start_date: str = '2024-01-01', end_date: str = '2024-12-31') -> Dict:
        """
        Calculate portfolio performance for a specific time period.
        
        Args:
            holdings (Dict): Portfolio holdings
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            Dict: Performance metrics for the period
        """
        # Get all equity and fixed income symbols
        equity_symbols = list(holdings['equity'].keys())
        fixed_income_symbols = list(holdings['fixed_income'].keys())
        all_symbols = equity_symbols + fixed_income_symbols
        
        # Get historical data for the period
        historical_data = {}
        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    historical_data[symbol] = data
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        if not historical_data:
            raise ValueError("No historical data available for the period")
        
        # Calculate daily returns for each asset
        returns_data = {}
        for symbol, data in historical_data.items():
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate portfolio weights
        weights = {}
        for symbol in equity_symbols:
            weights[symbol] = holdings['equity'][symbol]['weight']
        for symbol in fixed_income_symbols:
            weights[symbol] = holdings['fixed_income'][symbol]['weight']
        
        # Calculate portfolio returns
        portfolio_returns = returns_df.dot(pd.Series(weights))
        
        # Calculate initial portfolio value after commissions
        initial_value = self.portfolio_value
        total_commission = sum(holding.get('commission', 0) for holding in holdings['equity'].values())
        net_initial_value = initial_value - total_commission
        
        # Calculate cumulative portfolio value
        portfolio_values = (1 + portfolio_returns).cumprod() * net_initial_value
        
        # Calculate metrics
        total_return = (portfolio_values.iloc[-1] / net_initial_value) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility  # Assuming 2% risk-free rate
        
        # Calculate drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdowns = portfolio_values / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate monthly returns
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'monthly_returns': monthly_returns,
            'daily_returns': returns_df,  # Return the DataFrame instead of Series
            'portfolio_values': portfolio_values,
            'total_commission': total_commission,
            'net_initial_value': net_initial_value
        }

    def plot_portfolio_performance(self, performance: Dict, holdings: Dict, output_dir: str = "output/reports") -> None:
        """
        Create visualizations of portfolio performance.
        
        Args:
            performance (Dict): Performance metrics from calculate_forward_performance
            holdings (Dict): Portfolio holdings
            output_dir (str): Directory to save plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get S&P 500 data for the same period
        sp500 = yf.Ticker("^GSPC")
        sp500_data = sp500.history(start=performance['portfolio_values'].index[0], 
                                 end=performance['portfolio_values'].index[-1])
        sp500_values = (1 + sp500_data['Close'].pct_change().dropna()).cumprod() * performance['net_initial_value']
        
        # 1. Portfolio Value Over Time with S&P 500
        plt.figure(figsize=(12, 6))
        performance['portfolio_values'].plot(label='Portfolio')
        sp500_values.plot(label='S&P 500')
        plt.title('Portfolio Value vs S&P 500')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/portfolio_value_vs_sp500.png")
        plt.close()
        
        # 2. Monthly Returns Heatmap
        monthly_returns = performance['monthly_returns']
        monthly_returns_matrix = monthly_returns.values.reshape(-1, 1)
        months = [d.strftime('%b %Y') for d in monthly_returns.index]
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(monthly_returns_matrix, 
                   annot=True, 
                   fmt='.1%',
                   cmap='RdYlGn',
                   center=0,
                   yticklabels=months)
        plt.title('Monthly Returns Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monthly_returns_heatmap.png")
        plt.close()
        
        # 3. Drawdown Chart
        portfolio_values = performance['portfolio_values']
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values / rolling_max - 1) * 100
        
        plt.figure(figsize=(12, 6))
        drawdowns.plot()
        plt.title('Portfolio Drawdown Over Time')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/drawdown_chart.png")
        plt.close()
        
        # 4. Return Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(performance['daily_returns'].mean(axis=1) * 100, bins=50)  # Use mean of daily returns
        plt.title('Daily Returns Distribution')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/returns_distribution.png")
        plt.close()
        
        # 5. Sector Returns
        sector_stocks = {}
        for symbol, holding in holdings['equity'].items():
            if 'sector' in holding:
                sector = holding['sector']
                if sector not in sector_stocks:
                    sector_stocks[sector] = []
                sector_stocks[sector].append(symbol)
        
        # Calculate sector returns
        sector_performance = {}
        for sector, symbols in sector_stocks.items():
            sector_data = pd.DataFrame()
            for symbol in symbols:
                if symbol in performance['daily_returns'].columns:
                    sector_data[symbol] = performance['daily_returns'][symbol]
            if not sector_data.empty:
                # Calculate weighted returns for the sector
                weights = [holdings['equity'][symbol]['weight'] for symbol in symbols if symbol in performance['daily_returns'].columns]
                weights = np.array(weights) / sum(weights)  # Normalize weights
                sector_returns = sector_data.dot(weights)
                sector_performance[sector] = (1 + sector_returns).cumprod() * performance['net_initial_value']
        
        # Plot sector returns
        plt.figure(figsize=(12, 6))
        for sector, returns in sector_performance.items():
            returns.plot(label=sector)
        plt.title('Sector Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sector_performance.png")
        plt.close()
        
        # 6-8. Individual Stock Returns for each sector
        for sector, symbols in sector_stocks.items():
            plt.figure(figsize=(12, 6))
            for symbol in symbols:
                if symbol in performance['daily_returns'].columns:
                    stock_returns = (1 + performance['daily_returns'][symbol]).cumprod() * performance['net_initial_value']
                    stock_returns.plot(label=symbol)
            plt.title(f'{sector.title()} Stocks Performance')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{sector}_stocks_performance.png")
            plt.close()

    def optimize_stock_selection(self, lookback_years: int = 5, start_date: str = '2024-01-01', end_date: str = '2024-12-31') -> Dict:
        """
        Optimize stock selection and weights within sectors.
        Selects top 5 stocks from tech and finance, top 3 from energy,
        and uses ETFs for healthcare, consumer discretionary, and industrial.
        
        Args:
            lookback_years (int): Number of years of historical data to use for optimization
            start_date (str): Start date for forward performance calculation
            end_date (str): End date for forward performance calculation
            
        Returns:
            Dict: Optimized portfolio holdings
        """
        # Get historical data for all stocks and ETFs
        all_stocks = []
        for sector in ['technology', 'financials', 'energy']:
            all_stocks.extend(self.stock_universe[sector])
        
        # Add ETFs to the list
        all_stocks.extend(self.recommended_etfs.values())
        
        # Get historical data for optimization
        historical_data = self.get_portfolio_historical_data(all_stocks, period=f'{lookback_years}y')
        
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
            return -sharpe_ratio  # Negative because we're minimizing
        
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
                        'weight': self.asset_weights['fixed_income'] * 0.7,
                        'amount': self.portfolio_value * self.asset_weights['fixed_income'] * 0.7
                    },
                    'TLT': {
                        'name': 'iShares 20+ Year Treasury Bond ETF',
                        'weight': self.asset_weights['fixed_income'] * 0.3,
                        'amount': self.portfolio_value * self.asset_weights['fixed_income'] * 0.3
                    }
                },
                'cash': {
                    'USD': {
                        'name': 'Cash',
                        'weight': self.asset_weights['cash'],
                        'amount': self.portfolio_value * self.asset_weights['cash']
                    }
                }
            }
            
            # Add equity holdings
            for i, symbol in enumerate(selected_symbols):
                weight = result.x[i]
                amount = self.portfolio_value * weight
                
                if symbol in self.recommended_etfs.values():
                    # ETF holding
                    sector = next(s for s, etf in self.recommended_etfs.items() if etf == symbol)
                    holdings['equity'][symbol] = {
                        'name': f"{sector.title()} ETF",
                        'weight': weight,
                        'amount': amount,
                        'sector': sector
                    }
                else:
                    # Stock holding
                    stock_info = next((s for s in sum(stocks_info.values(), []) if s.symbol == symbol), None)
                    if stock_info:
                        shares = int(amount / stock_info.current_price)
                        commission = self.calculate_commission(shares, stock_info.current_price)
                        actual_amount = (shares * stock_info.current_price) + commission
                        
                        holdings['equity'][symbol] = {
                            'name': stock_info.name,
                            'weight': actual_amount / self.portfolio_value,
                            'amount': actual_amount,
                            'shares': shares,
                            'commission': commission,
                            'sector': stock_info.sector,
                            'score': stock_info.score
                        }
            
            # Calculate forward performance
            performance = self.calculate_forward_performance(holdings, start_date, end_date)
            
            # Print performance metrics
            print("\nForward Performance Metrics (2024):")
            print("--------------------------------")
            print(f"Initial Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Total Commission: ${performance['total_commission']:,.2f}")
            print(f"Net Initial Value: ${performance['net_initial_value']:,.2f}")
            print(f"Final Portfolio Value: ${performance['portfolio_values'].iloc[-1]:,.2f}")
            print(f"Total Return: {performance['total_return']:.1%}")
            print(f"Annualized Return: {performance['annualized_return']:.1%}")
            print(f"Volatility: {performance['volatility']:.1%}")
            print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {performance['max_drawdown']:.1%}")

            # Print ending value and weight of each position
            print("\nEnding Position Values and Weights:")
            print("----------------------------------")
            final_date = performance['portfolio_values'].index[-1]
            final_portfolio_value = performance['portfolio_values'].iloc[-1]
            # Combine equity and fixed income symbols
            all_symbols = list(holdings['equity'].keys()) + list(holdings['fixed_income'].keys())
            starting_positions = []
            ending_positions = []
            for symbol in all_symbols:
                # Get initial weight and amount
                if symbol in holdings['equity']:
                    initial_weight = holdings['equity'][symbol]['weight']
                    initial_amount = holdings['equity'][symbol]['amount']
                    name = holdings['equity'][symbol]['name']
                    asset_class = 'Equity'
                    sector = holdings['equity'][symbol].get('sector', '')
                else:
                    initial_weight = holdings['fixed_income'][symbol]['weight']
                    initial_amount = holdings['fixed_income'][symbol]['amount']
                    name = holdings['fixed_income'][symbol]['name']
                    asset_class = 'Fixed Income'
                    sector = 'Fixed Income'
                # Get ending value
                if symbol in performance['daily_returns'].columns:
                    # Calculate cumulative return for this asset
                    asset_returns = performance['daily_returns'][symbol]
                    asset_cum_return = (1 + asset_returns).cumprod()
                    ending_value = initial_amount * asset_cum_return.loc[final_date]
                else:
                    # If no return data, assume static value
                    ending_value = initial_amount
                ending_weight = ending_value / final_portfolio_value
                print(f"{symbol}: Value = ${ending_value:,.2f}, Weight = {ending_weight:.2%}")
                # Collect for CSV
                starting_positions.append({
                    'Symbol': symbol,
                    'Name': name,
                    'Asset Class': asset_class,
                    'Sector': sector,
                    'Start Value': initial_amount,
                    'Start Weight': initial_weight
                })
                ending_positions.append({
                    'Symbol': symbol,
                    'Name': name,
                    'Asset Class': asset_class,
                    'Sector': sector,
                    'End Value': ending_value,
                    'End Weight': ending_weight
                })
            # Export to CSV
            output_dir = "output/reports"
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            pd.DataFrame(starting_positions).to_csv(f"{output_dir}/starting_positions_{timestamp}.csv", index=False)
            pd.DataFrame(ending_positions).to_csv(f"{output_dir}/ending_positions_{timestamp}.csv", index=False)
            print(f"\nStarting and ending positions exported to {output_dir}/starting_positions_{timestamp}.csv and {output_dir}/ending_positions_{timestamp}.csv")

            # Create visualizations
            self.plot_portfolio_performance(performance, holdings)
            print("\nPerformance charts have been saved to the output/reports directory.")
            
            return holdings
        else:
            print("Optimization failed. Using current weights.")
            return self.get_target_holdings()


if __name__ == "__main__":
    # Example usage
    portfolio = Portfolio(initial_capital=300_000)
    
    # First optimize sector allocation using 5 years of data prior to 2023-01-01
    print("\nOptimizing sector allocation using data from 2018-01-01 to 2023-01-01...")
    optimized_sector_weights = portfolio.optimize_portfolio(lookback_years=5)
    
    # Then optimize stock selection as of 2023-01-01
    print("\nOptimizing stock selection as of 2023-01-01...")
    holdings = portfolio.optimize_stock_selection(
        lookback_years=5,
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    # Calculate metrics
    metrics = portfolio.calculate_portfolio_metrics(holdings)
    
    # Export to CSV
    portfolio.export_to_csv(holdings, metrics)
    
    # Print portfolio overview
    print("\nPortfolio Overview:")
    print("------------------")
    print(f"Total Value: ${metrics.total_value:,.2f}")
    print(f"\nAsset Allocation:")
    print(f"Equity: {metrics.equity_weight:.1%}")
    print(f"Fixed Income: {metrics.fixed_income_weight:.1%}")
    print(f"Cash: {metrics.cash_weight:.1%}")
    
    print("\nEquity Holdings:")
    print("---------------")
    for symbol, holding in holdings['equity'].items():
        print(f"\n{symbol} - {holding['name']}")
        print(f"  Weight: {holding['weight']:.1%}")
        print(f"  Amount: ${holding['amount']:,.2f}")
        if 'shares' in holding:
            print(f"  Shares: {holding['shares']}")
        if 'commission' in holding:
            print(f"  Commission: ${holding['commission']:,.2f}")
        if 'score' in holding:
            print(f"  Score: {holding['score']:.3f}")
    
    print("\nFixed Income Holdings:")
    print("---------------------")
    for symbol, holding in holdings['fixed_income'].items():
        print(f"\n{symbol} - {holding['name']}")
        print(f"  Weight: {holding['weight']:.1%}")
        print(f"  Amount: ${holding['amount']:,.2f}")
    
    print("\nRisk & Return Metrics:")
    print("---------------------")
    print(f"Expected Return: {metrics.expected_return:.1%}")
    print(f"Expected Volatility: {metrics.expected_volatility:.1%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {metrics.max_drawdown:.1%}")
    
    print("\nSector Weights:")
    print("--------------")
    for sector, weight in metrics.sector_weights.items():
        print(f"{sector}: {weight:.1%}")
    
    print("\nStyle Weights:")
    print("-------------")
    for style, weight in metrics.style_weights.items():
        print(f"{style}: {weight:.1%}")