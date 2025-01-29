class RegimeConfig:
    # These values need adjustment to make leverage more responsive
    REGIME_ADJ_FACTOR = 0.5     # Currently maps regime to 1.0-2.0 range
    TREND_ADJ_FACTOR = 0.3      # Currently maps trend to 1.0-1.6 range
    VOL_ADJ_FACTOR = 0.2        # Controls volatility impact
    MOMENTUM_ADJ_FACTOR = 1.2   # Maximum momentum boost

    DRAWDOWN_THRESHOLD = -0.15
    RECOVERY_THRESHOLD = -0.10