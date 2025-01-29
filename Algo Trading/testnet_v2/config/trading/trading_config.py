class TradingConfig:
    # Position and Risk Management
    MAX_POSITION_PCT = 0.3       # Maximum position size as percentage of portfolio
    MIN_POSITION_CHANGE = 0.05   # Minimum position change in one trade, avoid small trades
    MAX_POSITION_CHANGE = 0.2    # Maximum allowed position change in one trade
    
    # Trading Costs and Frequency
    COMMISSION_RATE = 0.00045    # Binance futures trading fee (0.045%)
    REBALANCE_THRESHOLD = 0.1    # Minimum weight deviation to trigger rebalance
    RISK_FREE_RATE = 0.02        # Risk-free rate for Sharpe ratio calculation
    COMMISSION_BUFFER_PERCENTAGE = 0.01  # Reserve 1% for commissions

    # Performance Calculation
    TRADING_DAYS = 252
    PERIODS_PER_DAY = 24        # For hourly data
    ANNUALIZATION_FACTOR = TRADING_DAYS * PERIODS_PER_DAY

    TIMEFRAME = '1h'

    SYMBOLS = [
        'ADAUSDT',
        'BNBUSDT',
        'BTCUSDT',
        'EOSUSDT',
        'ETHUSDT',
        'LTCUSDT',
        'NEOUSDT',
        'QTUMUSDT',
        'XRPUSDT'
    ]