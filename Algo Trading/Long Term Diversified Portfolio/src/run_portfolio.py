"""
Simple script to demonstrate basic portfolio functionality.
"""

from core.portfolio import Portfolio, Equity, FixedIncome
from data.fetcher import MarketDataFetcher


def main():
    # Initialize components
    portfolio = Portfolio(initial_capital=100000.0)
    fetcher = MarketDataFetcher()
    
    # Define our target holdings
    equity_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    fixed_income_symbols = ['TLT', 'AGG']
    
    # Get current prices and info
    all_symbols = equity_symbols + fixed_income_symbols
    prices = fetcher.get_current_prices(all_symbols)
    
    # Create equity holdings
    equities = []
    for symbol in equity_symbols:
        if symbol in prices:
            info = fetcher.get_basic_info(symbol)
            equity = Equity(
                symbol=symbol,
                name=info['name'],
                sector=info['sector'],
                market_cap=info['market_cap'],
                dividend_yield=info['dividend_yield']
            )
            equities.append(equity)
            portfolio.price_manager.update_price(symbol, prices[symbol])
    
    # Create fixed income holdings
    fixed_income = []
    for symbol in fixed_income_symbols:
        if symbol in prices:
            info = fetcher.get_basic_info(symbol)
            bond = FixedIncome(
                symbol=symbol,
                name=info['name'],
                coupon_rate=0.0,  # Would need to fetch this from a different source
                maturity_date='2025-12-31',  # Would need to fetch this from a different source
                credit_rating='AAA'  # Would need to fetch this from a different source
            )
            fixed_income.append(bond)
            portfolio.price_manager.update_price(symbol, prices[symbol])
    
    # Update portfolio holdings
    portfolio.update_holdings({
        'us_equities': equities,
        'fixed_income': fixed_income,
        'cash': 5000.0
    })
    
    # Print results
    print("\nPortfolio Summary:")
    print("-----------------")
    
    print("\nCurrent Allocation:")
    allocation = portfolio.get_current_allocation()
    for asset_class, percentage in allocation.items():
        print(f"{asset_class}: {percentage:.1%}")
    
    print("\nRebalancing Needs:")
    rebalancing = portfolio.calculate_rebalancing_needs()
    for asset_class, amount in rebalancing.items():
        print(f"{asset_class}: ${amount:,.2f}")


if __name__ == "__main__":
    main() 