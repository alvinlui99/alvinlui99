class Config:
    def __init__(self):
        self.coins = [
                "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT",
                "DOTUSDT","TRXUSDT","AVAXUSDT","ATOMUSDT",
                "LTCUSDT","XRPUSDT","UNIUSDT","AAVEUSDT","DOGEUSDT"
        ]

        # Data collection parameters
        self.lookback_days = 10
        self.interval = '1h'

        # Modelling parameters
        self.coint_pvalue_threshold = 0.05        
        self.window = self.lookback_days * 24 # 1 hour intervals

        # Copula-based strategy
        self.long_threshold = 0.95            # copula-based MI threshold for long positions
        self.long_exit_threshold = 0.5        # copula-based MI threshold for long exit
        self.short_threshold = 0.05           # copula-based MI threshold for short positions
        self.short_exit_threshold = 0.5       # copula-based MI threshold for short exit

        # Risk management
        self.investable_budget_pc = 0.95
        self.max_positions = 10
        self.take_profit_pc = 0.05
        self.stop_loss_pc = 0.02

        # Backtest parameters
        self.commission_pc = 0.00045 # 0.045%