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
        
        # Define asset class weights
        self.asset_weights = {
            'equity': 0.65,
            'fixed_income': 0.30,
            'cash': 0.05
        }
        
        # Define sector weights within equity
        self.sector_weights = {
            'technology': 0.28,          # 28% - Technology
            'healthcare': 0.13,          # 13% - Healthcare
            'financials': 0.12,          # 12% - Financials
            'consumer_discretionary': 0.10,  # 10% - Consumer Discretionary
            'industrials': 0.08,         # 8% - Industrials
            'consumer_staples': 0.07,    # 7% - Consumer Staples
            'energy': 0.05,             # 5% - Energy
            'materials': 0.05,          # 5% - Materials
            'utilities': 0.04,          # 4% - Utilities
            'real_estate': 0.04,        # 4% - Real Estate
            'communication_services': 0.04  # 4% - Communication Services
        }  # Total: 100%
        
        # Define scoring weights for each sector
        self.scoring_weights = {
            'technology': {
                'roic': 0.30,
                'profit_margin': 0.25,
                'market_cap': 0.20,
                'debt_to_equity': 0.15,
                'beta': 0.10
            },
            'healthcare': {
                'roic': 0.25,
                'profit_margin': 0.25,
                'market_cap': 0.20,
                'debt_to_equity': 0.15,
                'beta': 0.15
            },
            'financials': {
                'roic': 0.20,
                'profit_margin': 0.15,
                'market_cap': 0.20,
                'debt_to_equity': 0.15,
                'dividend_yield': 0.20,
                'beta': 0.10
            },
            'consumer_discretionary': {
                'roic': 0.25,
                'profit_margin': 0.20,
                'market_cap': 0.20,
                'debt_to_equity': 0.15,
                'beta': 0.20
            }
        }
        
        # Define stock universe
        self.stock_universe = {
            'technology': [
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AVGO', 'ASML', 'CRM', 'ADBE',
                'CSCO', 'INTC', 'AMD', 'QCOM', 'TXN', 'MU', 'AMAT'
            ],
            'healthcare': [
                'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'LLY', 'DHR',
                'BMY', 'AMGN', 'GILD', 'ISRG', 'VRTX', 'REGN', 'CI'
            ],
            'financials': [
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP',
                'V', 'MA', 'C', 'USB', 'PNC', 'TFC', 'MMC'
            ],
            'consumer_discretionary': [
                'AMZN', 'MCD', 'SBUX', 'NKE', 'HD', 'LOW', 'DIS', 'NFLX',
                'TSLA', 'MAR', 'BKNG', 'CMG', 'TGT', 'COST', 'TJX'
            ]
        }
        
        # Define recommended ETFs
        self.recommended_etfs = {
            'mid_cap': 'IJH',  # iShares Core S&P Mid-Cap ETF
            'small_cap': 'IJR',  # iShares Core S&P Small-Cap ETF
            'industrials': 'XLI',  # Industrial Select Sector SPDR Fund
            'consumer_staples': 'XLP',  # Consumer Staples Select Sector SPDR Fund
            'energy': 'XLE',  # Energy Select Sector SPDR Fund
            'materials': 'XLB',  # Materials Select Sector SPDR Fund
            'utilities': 'XLU',  # Utilities Select Sector SPDR Fund
            'real_estate': 'XLRE',  # Real Estate Select Sector SPDR Fund
            'communication_services': 'XLC'  # Communication Services Select Sector SPDR Fund
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
        
        # Handle missing or invalid values
        roic = max(0, stock.roic)
        profit_margin = max(0, stock.profit_margin)
        market_cap = max(0, stock.market_cap)
        debt_to_equity = min(10, max(0, stock.debt_to_equity))
        dividend_yield = max(0, stock.dividend_yield)
        beta = min(3, max(0, stock.beta))
        
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
            holdings.append({
                'symbol': stock.symbol,
                'name': stock.name,
                'weight': position_size / self.portfolio_value,
                'amount': position_size,
                'shares': int(position_size / stock.current_price),
                'sector': sector,
                'score': stock.score
            })
        
        return holdings
    
    def get_target_holdings(self) -> Dict:
        """Get target holdings for the portfolio."""
        # Calculate asset class allocations
        equity_value = self.portfolio_value * self.asset_weights['equity']
        fixed_income_value = self.portfolio_value * self.asset_weights['fixed_income']
        cash_value = self.portfolio_value * self.asset_weights['cash']
        
        # Get equity holdings
        equity_holdings = {}
        for sector, weight in self.sector_weights.items():
            if sector in ['technology', 'healthcare', 'financials', 'consumer_discretionary']:
                # Direct holdings
                sector_value = equity_value * weight
                stocks = self.get_recommended_stocks(sector, sector_value)
                for stock in stocks:
                    equity_holdings[stock['symbol']] = stock
            else:
                # ETF holdings
                etf_symbol = self.recommended_etfs.get(sector)
                if etf_symbol:
                    equity_holdings[etf_symbol] = {
                        'name': f"{sector.title()} ETF",
                        'weight': (equity_value * weight) / self.portfolio_value,
                        'amount': equity_value * weight,
                        'sector': sector
                    }
        
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
        """Calculate portfolio-level metrics using historical data."""
        # Calculate weights
        equity_weight = sum(holding['weight'] for holding in holdings['equity'].values())
        fixed_income_weight = sum(holding['weight'] for holding in holdings['fixed_income'].values())
        cash_weight = holdings['cash']['USD']['weight']
        
        # Calculate sector weights
        sector_weights = {}
        for holding in holdings['equity'].values():
            sector = holding['sector']
            sector_weights[sector] = sector_weights.get(sector, 0) + holding['weight']
        
        # Calculate style weights (simplified)
        style_weights = {
            'large_cap': 0.50,  # Assuming 50% of equity is large cap
            'mid_cap': 0.25,    # From IJH
            'small_cap': 0.25   # From IJR
        }
        
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
        
        # Calculate correlation matrices
        full_corr = returns_df.corr()
        
        # Calculate within-sector correlations
        sector_correlations = {}
        for sector in self.sector_weights.keys():
            sector_symbols = [symbol for symbol in equity_symbols 
                            if holdings['equity'][symbol]['sector'] == sector]
            if len(sector_symbols) > 1:  # Only calculate if we have multiple stocks in sector
                sector_corr = returns_df[sector_symbols].corr()
                sector_correlations[sector] = {
                    'mean_correlation': sector_corr.values[np.triu_indices_from(sector_corr.values, k=1)].mean(),
                    'max_correlation': sector_corr.values[np.triu_indices_from(sector_corr.values, k=1)].max(),
                    'min_correlation': sector_corr.values[np.triu_indices_from(sector_corr.values, k=1)].min()
                }
        
        # Calculate cross-sector correlations
        cross_sector_correlations = {}
        sectors = list(self.sector_weights.keys())
        for i, sector1 in enumerate(sectors):
            for sector2 in sectors[i+1:]:
                sector1_symbols = [symbol for symbol in equity_symbols 
                                 if holdings['equity'][symbol]['sector'] == sector1]
                sector2_symbols = [symbol for symbol in equity_symbols 
                                 if holdings['equity'][symbol]['sector'] == sector2]
                if sector1_symbols and sector2_symbols:
                    cross_corr = returns_df[sector1_symbols + sector2_symbols].corr()
                    # Get correlations between sectors
                    sector1_indices = range(len(sector1_symbols))
                    sector2_indices = range(len(sector1_symbols), len(sector1_symbols) + len(sector2_symbols))
                    cross_sector_corr = cross_corr.iloc[sector1_indices, sector2_indices]
                    cross_sector_correlations[f"{sector1}-{sector2}"] = {
                        'mean_correlation': cross_sector_corr.values.mean(),
                        'max_correlation': cross_sector_corr.values.max(),
                        'min_correlation': cross_sector_corr.values.min()
                    }
        
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
        
        # Print correlation analysis
        print("\nCorrelation Analysis:")
        print("-------------------")
        print("\nWithin-Sector Correlations:")
        for sector, corr_stats in sector_correlations.items():
            print(f"\n{sector.title()}:")
            print(f"  Mean Correlation: {corr_stats['mean_correlation']:.3f}")
            print(f"  Max Correlation: {corr_stats['max_correlation']:.3f}")
            print(f"  Min Correlation: {corr_stats['min_correlation']:.3f}")
        
        print("\nCross-Sector Correlations:")
        for sectors_pair, corr_stats in cross_sector_correlations.items():
            print(f"\n{sectors_pair}:")
            print(f"  Mean Correlation: {corr_stats['mean_correlation']:.3f}")
            print(f"  Max Correlation: {corr_stats['max_correlation']:.3f}")
            print(f"  Min Correlation: {corr_stats['min_correlation']:.3f}")
        
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
            style_weights=style_weights
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


if __name__ == "__main__":
    # Example usage
    portfolio = Portfolio(initial_capital=1_000_000)
    
    # Get target holdings
    holdings = portfolio.get_target_holdings()
    
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
    print(f"Estimated Max Drawdown: {metrics.max_drawdown:.1%}")
    
    print("\nSector Weights:")
    print("--------------")
    for sector, weight in metrics.sector_weights.items():
        print(f"{sector}: {weight:.1%}")
    
    print("\nStyle Weights:")
    print("-------------")
    for style, weight in metrics.style_weights.items():
        print(f"{style}: {weight:.1%}")