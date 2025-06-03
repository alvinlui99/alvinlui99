import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import os
import logging
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS

from data import BinanceDataCollector
from utils import PairIndicators

logger = logging.getLogger(__name__)

def run_weekly_coint_tests(start_date: str, end_date: str, d: int) -> pd.DataFrame:
    data_collector = BinanceDataCollector()
    
    # Convert string dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    results = []
    
    # Generate weekly windows
    current_start = start_dt
    while current_start + timedelta(days=d) <= end_dt:
        current_end = current_start + timedelta(days=d)
        
        # Get data for BTCUSDT and ETHUSDT
        btc_data = data_collector.get_historical_klines('BTCUSDT', '1h', 
                                                       current_start.strftime('%Y-%m-%d %H:%M:%S'),
                                                       current_end.strftime('%Y-%m-%d %H:%M:%S'))
        eth_data = data_collector.get_historical_klines('ETHUSDT', '1h',
                                                       current_start.strftime('%Y-%m-%d %H:%M:%S'),
                                                       current_end.strftime('%Y-%m-%d %H:%M:%S'))

        # Align dataframes
        btc_data, eth_data = PairIndicators.align_dataframes(btc_data, eth_data)

        beta = OLS(btc_data['close'].values, eth_data['close'].values).fit().params[0]
        spread = btc_data['close'] - beta * eth_data['close']
        result = adfuller(spread)
        adf_stat = result[0]
        p_value = result[1]
        
        results.append({
            'start_date': current_start.strftime('%Y-%m-%d'),
            'end_date': current_end.strftime('%Y-%m-%d'),
            'beta': beta,
            'adf_stat': adf_stat,
            'p_value': p_value
        })
        
        current_start += timedelta(days=1)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    d = 3
    start_date = '2020-01-01 00:00:00'
    end_date = '2024-12-31 23:59:59'
    
    # Run cointegration tests
    results_df = run_weekly_coint_tests(start_date, end_date, d)
    
    # Export results to CSV
    output_path = f'cointegration_results_2024_d_{d}.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"d: {d}")
    print(f"Total number of tests: {len(results_df)}")
    print(f"Number of significant cointegration (p < 0.05): {len(results_df[results_df['p_value'] < 0.05])}")
    print(f"Average p-value: {results_df['p_value'].mean():.4f}")