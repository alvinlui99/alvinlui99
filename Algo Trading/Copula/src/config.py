class Config:
    def __init__(self):
        # Data collection parameters
        self.lookback_days = 10
        self.interval = '1h'

        # Modelling parameters
        self.coint_pvalue_threshold = 0.05        
        self.window = self.lookback_days * 24 # 1 hour intervals

        # Copula-based strategy
        self.long_threshold = 0.95            # copula-based MI threshold for long positions
        self.short_threshold = 0.05           # copula-based MI threshold for short positions

        # Risk management
        self.investable_budget_pc = 0.95
        self.max_positions = 5