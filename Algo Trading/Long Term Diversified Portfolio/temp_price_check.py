import yfinance as yf
import pandas as pd

# List of symbols to check
symbols = ['ENPH', 'SEDG', 'ICLN', 'TAN', 'QCLN']

# Date to check
start_date = '2017-12-28'
end_date = '2018-01-01'

# Get data for each symbol
for symbol in symbols:
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if not data.empty:
            price = data['Close'].iloc[0]
            print(f"{symbol}: ${price:.2f}")
        else:
            print(f"{symbol}: No data available")
    except Exception as e:
        print(f"{symbol}: Error - {str(e)}") 