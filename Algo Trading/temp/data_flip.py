import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def get_decimal_places(series):
    """Get the maximum number of decimal places in a series"""
    # Convert to string and get the part after decimal point
    decimals = series.astype(str).str.split('.').str[1]
    # Return the maximum length, default to 0 if no decimals
    return decimals.str.len().max() if not decimals.isna().all() else 0

def flip_data(symbol, df):
    """
    Flip OHLC price data upside down.
    The global maximum becomes the global minimum in the flipped version.
    Maintains the original decimal precision for each column.
    """
    # Store original decimal places
    decimal_places = {
        'Open': get_decimal_places(df['Open']),
        'High': get_decimal_places(df['High']),
        'Low': get_decimal_places(df['Low']),
        'Close': get_decimal_places(df['Close'])
    }
    
    # Find global max and min across all OHLC values
    global_max = max(df[['Open', 'High', 'Low', 'Close']].max())
    global_min = min(df[['Open', 'High', 'Low', 'Close']].min())
    
    # Create new dataframe with flipped data
    flipped_df = df.copy()
    
    # Flip each price by: flipped = max + min - original
    # This ensures the global max becomes global min and vice versa
    flipped_df['High'] = (global_max + global_min - df['Low']).round(decimal_places['High'])
    flipped_df['Low'] = (global_max + global_min - df['High']).round(decimal_places['Low'])
    flipped_df['Open'] = (global_max + global_min - df['Open']).round(decimal_places['Open'])
    flipped_df['Close'] = (global_max + global_min - df['Close']).round(decimal_places['Close'])
    
    # Validate OHLC relationships
    invalid_rows = (
        (flipped_df['Low'] > flipped_df['High']) |
        (flipped_df['Low'] > flipped_df['Open']) |
        (flipped_df['Low'] > flipped_df['Close']) |
        (flipped_df['High'] < flipped_df['Open']) |
        (flipped_df['High'] < flipped_df['Close'])
    )
    
    if invalid_rows.any():
        print(f"Warning: Found {invalid_rows.sum()} rows with invalid OHLC relationships for {symbol}")
        print("Last invalid row:")
        print(flipped_df[invalid_rows].iloc[0])
        raise ValueError(f"Invalid OHLC relationships detected after flipping for {symbol}")
    
    return flipped_df

def plot_comparison(original_df, flipped_df, symbol):
    """Plot original and flipped data for comparison"""
    plt.figure(figsize=(15, 7))
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in original_df.columns:
        original_df['timestamp'] = pd.to_datetime(original_df['timestamp'], unit='ms')
        flipped_df['timestamp'] = pd.to_datetime(flipped_df['timestamp'], unit='ms')
        x_axis = 'timestamp'
    else:
        x_axis = original_df.index

    # Plot original data
    plt.plot(x_axis, original_df['Close'], label='Original', color='blue', alpha=0.7)
    
    # Plot flipped data
    plt.plot(x_axis, flipped_df['Close'], label='Flipped', color='red', alpha=0.7)
    
    plt.title(f'Original vs Flipped Price Data - {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plots_dir = 'comparison_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(os.path.join(plots_dir, f'{symbol}_comparison.png'))
    plt.close()

def process_directory():
    # Create 1h_flip directory if it doesn't exist
    if not os.path.exists('1h_flip'):
        os.makedirs('1h_flip')
    
    # Process all CSV files in the 1h directory
    for filename in os.listdir('1h'):
        if filename.endswith('.csv'):
            symbol = filename.replace('.csv', '')
            
            # Read the input file
            input_path = os.path.join('1h', filename)
            df = pd.read_csv(input_path)
            
            # Flip the data
            flipped_df = flip_data(symbol, df)
            
            # Plot comparison
            plot_comparison(df, flipped_df, symbol)
            
            # Save to output directory
            output_path = os.path.join('1h_flip', filename)
            flipped_df.to_csv(output_path, index=False)
            print(f"Processed and plotted {filename}")

if __name__ == "__main__":
    process_directory()
