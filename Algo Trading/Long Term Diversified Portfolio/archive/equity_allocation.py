"""
Equity allocation strategy module - optimized for personal portfolio management.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf


@dataclass
class EquityStyle:
    """Represents an equity style category."""
    name: str
    target_weight: float
    description: str
    implementation: str = 'Direct'  # Default to direct implementation


@dataclass
class SectorAllocation:
    """Represents a sector allocation target."""
    name: str
    target_weight: float
    implementation: str = 'Direct'  # Default to direct implementation
    min_weight: float = 0.0
    max_weight: float = 1.0


class EquityAllocationStrategy:
    """Manages equity allocation strategy and target weights."""
    
    def __init__(self):
        # Define style allocations - using ETFs for small/mid cap exposure
        self.styles = {
            'large_growth': EquityStyle('Large Growth', 0.25, 'Large-cap growth stocks', 'Direct'),
            'large_value': EquityStyle('Large Value', 0.25, 'Large-cap value stocks', 'Direct'),
            'mid_cap': EquityStyle('Mid Cap', 0.25, 'Mid-cap stocks', 'ETF'),  # Using ETF for mid-cap
            'small_cap': EquityStyle('Small Cap', 0.25, 'Small-cap stocks', 'ETF')  # Using ETF for small-cap
        }
        
        # Define sector allocations - using ETFs for less liquid sectors
        self.sectors = {
            'technology': SectorAllocation('Technology', 0.28, 0.20, 0.35, 'Direct'),
            'healthcare': SectorAllocation('Healthcare', 0.13, 0.08, 0.18, 'Direct'),
            'financials': SectorAllocation('Financials', 0.12, 0.08, 0.16, 'Direct'),
            'consumer_discretionary': SectorAllocation('Consumer Discretionary', 0.10, 0.06, 0.14, 'Direct'),
            'industrials': SectorAllocation('Industrials', 0.08, 0.05, 0.11, 'ETF'),
            'consumer_staples': SectorAllocation('Consumer Staples', 0.07, 0.04, 0.10, 'ETF'),
            'energy': SectorAllocation('Energy', 0.05, 0.03, 0.07, 'ETF'),
            'materials': SectorAllocation('Materials', 0.05, 0.03, 0.07, 'ETF'),
            'utilities': SectorAllocation('Utilities', 0.03, 0.02, 0.04, 'ETF'),
            'real_estate': SectorAllocation('Real Estate', 0.03, 0.02, 0.04, 'ETF'),
            'communication_services': SectorAllocation('Communication Services', 0.03, 0.02, 0.04, 'ETF')
        }
        
        # Define quality metrics thresholds
        self.quality_metrics = {
            'min_market_cap': 1e9,  # $1B minimum market cap
            'min_profit_margin': 0.10,  # 10% minimum profit margin
            'max_debt_to_equity': 1.0,  # Maximum debt-to-equity ratio
            'min_roic': 0.12  # Minimum return on invested capital
        }
        
        # Define recommended ETFs for each style/sector
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
    
    def get_style_allocation(self) -> Dict[str, float]:
        """Get the target allocation by style."""
        return {style.name: style.target_weight for style in self.styles.values()}
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """Get the target allocation by sector."""
        return {sector.name: sector.target_weight for sector in self.sectors.values()}
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """Get the quality metrics thresholds."""
        return self.quality_metrics
    
    def get_target_holdings(self, portfolio_value: float) -> Dict[str, float]:
        """
        Calculate target holdings based on portfolio value.
        
        Args:
            portfolio_value (float): Total portfolio value
            
        Returns:
            Dict[str, float]: Target holdings by style and sector
        """
        equity_value = portfolio_value * 0.65  # 65% to equities per IPS
        
        # Calculate style targets
        style_targets = {
            style: weight * equity_value 
            for style, weight in self.get_style_allocation().items()
        }
        
        # Calculate sector targets
        sector_targets = {
            sector: weight * equity_value 
            for sector, weight in self.get_sector_allocation().items()
        }
        
        return {
            'style_targets': style_targets,
            'sector_targets': sector_targets
        }
    
    def get_style_constraints(self) -> Dict[str, Dict[str, float]]:
        """Get the style constraints for portfolio construction."""
        return {
            'large_growth': {'min_market_cap': 10e9, 'max_market_cap': float('inf')},
            'large_value': {'min_market_cap': 10e9, 'max_market_cap': float('inf')},
            'mid_cap': {'min_market_cap': 2e9, 'max_market_cap': 10e9},
            'small_cap': {'min_market_cap': 300e6, 'max_market_cap': 2e9}
        }
    
    def get_implementation_plan(self, portfolio_value: float) -> Dict:
        """
        Get a practical implementation plan for the portfolio.
        
        Args:
            portfolio_value (float): Total portfolio value
            
        Returns:
            Dict: Implementation plan with direct holdings and ETFs
        """
        equity_value = portfolio_value * 0.65
        
        # Calculate ETF allocations
        etf_allocations = {}
        for style, style_info in self.styles.items():
            if style_info.implementation == 'ETF':
                etf_symbol = self.recommended_etfs.get(style)
                if etf_symbol:
                    etf_allocations[etf_symbol] = style_info.target_weight * equity_value
        
        for sector, sector_info in self.sectors.items():
            if sector_info.implementation == 'ETF':
                etf_symbol = self.recommended_etfs.get(sector)
                if etf_symbol:
                    etf_allocations[etf_symbol] = sector_info.target_weight * equity_value
        
        # Calculate direct stock allocation
        direct_allocation = equity_value - sum(etf_allocations.values())
        
        return {
            'etf_allocations': etf_allocations,
            'direct_stock_allocation': direct_allocation,
            'recommended_etfs': self.recommended_etfs
        }


if __name__ == "__main__":
    # Example usage
    strategy = EquityAllocationStrategy()
    
    # Print style allocation
    print("\nStyle Allocation:")
    for style, weight in strategy.get_style_allocation().items():
        print(f"{style}: {weight:.1%}")
    
    # Print sector allocation
    print("\nSector Allocation:")
    for sector, weight in strategy.get_sector_allocation().items():
        print(f"{sector}: {weight:.1%}")
    
    # Print implementation plan for $1M portfolio
    print("\nImplementation Plan for $1M Portfolio:")
    plan = strategy.get_implementation_plan(1_000_000)
    
    print("\nETF Allocations:")
    for etf, amount in plan['etf_allocations'].items():
        print(f"{etf}: ${amount:,.2f}")
    
    print(f"\nDirect Stock Allocation: ${plan['direct_stock_allocation']:,.2f}")
    
    print("\nRecommended ETFs:")
    for category, etf in plan['recommended_etfs'].items():
        print(f"{category}: {etf}") 