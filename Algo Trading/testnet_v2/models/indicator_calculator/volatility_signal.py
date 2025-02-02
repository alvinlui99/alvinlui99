from typing import Dict
import pandas as pd
import numpy as np
from .indicator_calculator import SignalCalculator
from config import RegimeConfig

class VolatilitySignalCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        indicator_calculators = {
            'historical': HistoricalVolatilityCalculator(),
            'garch': GarchVolatilityCalculator(),
            'parkinson': ParkinsonVolatilityCalculator(),
            'yang_zhang': YangZhangVolatilityCalculator()
        }
        signals = {
            name: calculator.calculate_indicators(indicators)
            for name, calculator in indicator_calculators.items()
        }
        weighted_signal = sum(signals[name] * RegimeConfig.VolatilitySignalConfig.WEIGHTS[name] for name in signals)
        return max(min(weighted_signal, 1.0), -1.0)

class HistoricalVolatilityCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate Historical Volatility signal using multiple timeframes and methods.
        Returns a value between -1 (decreasing volatility) and 1 (increasing volatility).
        
        Signal components:
        1. Multi-timeframe volatility trend
        2. Volatility regime changes
        3. Return distribution analysis
        4. Volatility term structure
        """
        close = indicators['close']
        returns = close.pct_change()
        
        # 1. Multi-timeframe Volatility Trend (-1 to 1)
        vol_signals = []
        for period in RegimeConfig.HistoricalVolConfig.PERIODS:
            vol = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
            vol_sma = vol.rolling(window=RegimeConfig.HistoricalVolConfig.TREND_PERIOD).mean()
            vol_signal = 0.0 if vol_sma.iloc[-1] == 0 else (vol.iloc[-1] - vol_sma.iloc[-1]) / vol_sma.iloc[-1]
            vol_signals.append(np.tanh(vol_signal * RegimeConfig.HistoricalVolConfig.TREND_FACTOR))
        
        trend_signal = np.average(vol_signals, weights=RegimeConfig.HistoricalVolConfig.PERIOD_WEIGHTS)
        
        # 2. Regime Change Detection (-1 to 1)
        regime_signal = self._detect_regime_change(returns)
        
        # 3. Return Distribution Analysis (-1 to 1)
        distribution_signal = self._analyze_return_distribution(returns)
        
        # 4. Volatility Term Structure (-1 to 1)
        term_structure_signal = self._calculate_term_structure(returns)
        
        # Combine signals
        composite_signal = (
            RegimeConfig.HistoricalVolConfig.WEIGHTS['trend'] * trend_signal +
            RegimeConfig.HistoricalVolConfig.WEIGHTS['regime'] * regime_signal +
            RegimeConfig.HistoricalVolConfig.WEIGHTS['distribution'] * distribution_signal +
            RegimeConfig.HistoricalVolConfig.WEIGHTS['term_structure'] * term_structure_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
    def _detect_regime_change(self, returns: pd.Series) -> float:
        """Detect volatility regime changes using rolling statistics."""
        short_vol = returns.rolling(window=RegimeConfig.HistoricalVolConfig.SHORT_WINDOW).std()
        long_vol = returns.rolling(window=RegimeConfig.HistoricalVolConfig.LONG_WINDOW).std()
        
        regime_ratio = 0.0 if long_vol.iloc[-1] == 0 else short_vol.iloc[-1] / long_vol.iloc[-1]
        return np.tanh((regime_ratio - 1) * RegimeConfig.HistoricalVolConfig.REGIME_FACTOR)
    
    def _analyze_return_distribution(self, returns: pd.Series) -> float:
        """Analyze return distribution characteristics."""
        window = returns.iloc[-RegimeConfig.HistoricalVolConfig.DISTRIBUTION_WINDOW:]
        
        skew = window.skew()
        kurt = window.kurtosis()
        
        distribution_score = (skew * RegimeConfig.HistoricalVolConfig.SKEW_WEIGHT +
                            (kurt - 3) * RegimeConfig.HistoricalVolConfig.KURT_WEIGHT)
        
        return np.tanh(distribution_score)
    
    def _calculate_term_structure(self, returns: pd.Series) -> float:
        """Calculate volatility term structure slope."""
        vols = []
        for period in RegimeConfig.HistoricalVolConfig.TERM_STRUCTURE_PERIODS:
            vol = returns.rolling(window=period).std() * np.sqrt(252)
            vols.append(vol.iloc[-1])
            
        slope = np.polyfit(range(len(vols)), vols, 1)[0]
        return np.tanh(slope * RegimeConfig.HistoricalVolConfig.TERM_STRUCTURE_FACTOR)

class GarchVolatilityCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate GARCH-based volatility signal.
        Returns a value between -1 (decreasing volatility) and 1 (increasing volatility).
        
        Signal components:
        1. GARCH volatility trend
        2. Volatility persistence
        3. Shock impact
        4. Forecast trend
        """
        returns = indicators['close'].pct_change()
        
        # Calculate GARCH components using rolling windows to approximate GARCH behavior
        squared_returns = returns ** 2
        garch_vol = self._calculate_rolling_garch(returns, squared_returns)
        
        # 1. GARCH Volatility Trend (-1 to 1)
        vol_sma = garch_vol.rolling(window=RegimeConfig.GarchConfig.TREND_PERIOD).mean()
        trend_signal = np.tanh((garch_vol.iloc[-1] - vol_sma.iloc[-1]) / vol_sma.iloc[-1] * 
                              RegimeConfig.GarchConfig.TREND_FACTOR)
        
        # 2. Volatility Persistence (-1 to 1)
        persistence_signal = self._calculate_persistence(garch_vol)
        
        # 3. Shock Impact (-1 to 1)
        shock_signal = self._calculate_shock_impact(returns, garch_vol)
        
        # 4. Forecast Trend (-1 to 1)
        forecast_signal = self._calculate_forecast_trend(garch_vol)
        
        # Combine signals
        composite_signal = (
            RegimeConfig.GarchConfig.WEIGHTS['trend'] * trend_signal +
            RegimeConfig.GarchConfig.WEIGHTS['persistence'] * persistence_signal +
            RegimeConfig.GarchConfig.WEIGHTS['shock'] * shock_signal +
            RegimeConfig.GarchConfig.WEIGHTS['forecast'] * forecast_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
    def _calculate_rolling_garch(self, returns: pd.Series, squared_returns: pd.Series) -> pd.Series:
        """Approximate GARCH(1,1) using rolling windows."""
        omega = RegimeConfig.GarchConfig.OMEGA
        alpha = RegimeConfig.GarchConfig.ALPHA
        beta = RegimeConfig.GarchConfig.BETA
        
        # Initialize with long-term variance
        lt_var = squared_returns.rolling(window=RegimeConfig.GarchConfig.LT_WINDOW).mean()
        
        # Calculate GARCH variance recursively
        garch_var = lt_var.copy()
        for i in range(1, len(returns)):
            garch_var.iloc[i] = (omega + 
                                alpha * squared_returns.iloc[i-1] + 
                                beta * garch_var.iloc[i-1])
        
        return np.sqrt(garch_var)
    
    def _calculate_persistence(self, garch_vol: pd.Series) -> float:
        """Calculate volatility persistence."""
        autocorr = garch_vol.autocorr(lag=RegimeConfig.GarchConfig.PERSISTENCE_LAG)
        return np.tanh(autocorr * RegimeConfig.GarchConfig.PERSISTENCE_FACTOR)
    
    def _calculate_shock_impact(self, returns: pd.Series, garch_vol: pd.Series) -> float:
        """Calculate impact of recent shocks on volatility."""
        recent_shocks = (returns.iloc[-RegimeConfig.GarchConfig.SHOCK_WINDOW:] / garch_vol.iloc[-RegimeConfig.GarchConfig.SHOCK_WINDOW:]).abs()
        shock_impact = recent_shocks.mean()
        return np.tanh(shock_impact * RegimeConfig.GarchConfig.SHOCK_FACTOR)
    
    def _calculate_forecast_trend(self, garch_vol: pd.Series) -> float:
        """Calculate forecasted volatility trend."""
        forecast_window = garch_vol.iloc[-RegimeConfig.GarchConfig.FORECAST_WINDOW:]
        slope = np.polyfit(range(len(forecast_window)), forecast_window, 1)[0]
        return np.tanh(slope * RegimeConfig.GarchConfig.FORECAST_FACTOR)

class ParkinsonVolatilityCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate Parkinson volatility signal using high-low price ranges.
        Returns a value between -1 (decreasing volatility) and 1 (increasing volatility).
        
        Signal components:
        1. Parkinson volatility trend
        2. Range expansion/contraction
        3. Relative range analysis
        4. Multi-timeframe comparison
        """
        high = indicators['high']
        low = indicators['low']
        
        # Calculate Parkinson volatility
        hl_ratio = np.log(high / low)
        parkinson_vol = np.sqrt(1 / (4 * np.log(2)) * hl_ratio ** 2)
        
        # 1. Volatility Trend (-1 to 1)
        vol_sma = parkinson_vol.rolling(window=RegimeConfig.ParkinsonConfig.TREND_PERIOD).mean()
        trend_signal = 0.0 if vol_sma.iloc[-1] == 0 else np.tanh((parkinson_vol.iloc[-1] - vol_sma.iloc[-1]) / vol_sma.iloc[-1] * 
                              RegimeConfig.ParkinsonConfig.TREND_FACTOR)
        
        # 2. Range Analysis (-1 to 1)
        range_signal = self._analyze_range_expansion(high, low)
        
        # 3. Relative Range (-1 to 1)
        relative_signal = self._calculate_relative_range(high, low)
        
        # 4. Multi-timeframe Analysis (-1 to 1)
        mtf_signal = self._calculate_mtf_signal(parkinson_vol)
        
        # Combine signals
        composite_signal = (
            RegimeConfig.ParkinsonConfig.WEIGHTS['trend'] * trend_signal +
            RegimeConfig.ParkinsonConfig.WEIGHTS['range'] * range_signal +
            RegimeConfig.ParkinsonConfig.WEIGHTS['relative'] * relative_signal +
            RegimeConfig.ParkinsonConfig.WEIGHTS['mtf'] * mtf_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
    def _analyze_range_expansion(self, high: pd.Series, low: pd.Series) -> float:
        """Analyze high-low range expansion/contraction."""
        ranges = high - low
        range_sma = ranges.rolling(window=RegimeConfig.ParkinsonConfig.RANGE_PERIOD).mean()
        expansion_rate = 0.0 if range_sma.iloc[-1] == 0 else (ranges.iloc[-1] - range_sma.iloc[-1]) / range_sma.iloc[-1]
        return np.tanh(expansion_rate * RegimeConfig.ParkinsonConfig.RANGE_FACTOR)
    
    def _calculate_relative_range(self, high: pd.Series, low: pd.Series) -> float:
        """Calculate relative range compared to historical ranges."""
        ranges = high - low
        percentile = (ranges <= ranges.iloc[-1]).mean()
        return (percentile - 0.5) * 2  # Convert to -1 to 1
    
    def _calculate_mtf_signal(self, parkinson_vol: pd.Series) -> float:
        """Calculate multi-timeframe volatility signal."""
        signals = []
        for period in RegimeConfig.ParkinsonConfig.MTF_PERIODS:
            vol_ma = parkinson_vol.rolling(window=period).mean()
            signal = 0.0 if vol_ma.iloc[-1] == 0 else (parkinson_vol.iloc[-1] - vol_ma.iloc[-1]) / vol_ma.iloc[-1]
            signals.append(np.tanh(signal * RegimeConfig.ParkinsonConfig.MTF_FACTOR))
        
        return np.average(signals, weights=RegimeConfig.ParkinsonConfig.MTF_WEIGHTS)

class YangZhangVolatilityCalculator(SignalCalculator):
    def calculate_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate Yang-Zhang volatility signal using open-high-low-close prices.
        Returns a value between -1 (decreasing volatility) and 1 (increasing volatility).
        
        Signal components:
        1. Yang-Zhang volatility trend
        2. Overnight gap analysis
        3. Intraday volatility
        4. Component comparison
        """
        open_price = indicators['open']
        high = indicators['high']
        low = indicators['low']
        close = indicators['close']
        
        # Calculate Yang-Zhang components
        overnight_vol = self._calculate_overnight_volatility(close, open_price)
        open_close_vol = self._calculate_open_close_volatility(open_price, close)
        rogers_satchell_vol = self._calculate_rogers_satchell_volatility(open_price, high, low, close)
        
        # Combine into Yang-Zhang volatility
        yz_vol = np.sqrt(overnight_vol + RegimeConfig.YangZhangConfig.K * open_close_vol + 
                        (1 - RegimeConfig.YangZhangConfig.K) * rogers_satchell_vol)
        
        # 1. Volatility Trend (-1 to 1)
        vol_sma = yz_vol.rolling(window=RegimeConfig.YangZhangConfig.TREND_PERIOD).mean()
        trend_signal = 0.0 if vol_sma.iloc[-1] == 0 else np.tanh((yz_vol.iloc[-1] - vol_sma.iloc[-1]) / vol_sma.iloc[-1] * 
                              RegimeConfig.YangZhangConfig.TREND_FACTOR)
        
        # 2. Overnight Analysis (-1 to 1)
        overnight_signal = self._analyze_overnight_gaps(close, open_price)
        
        # 3. Intraday Volatility (-1 to 1)
        intraday_signal = self._analyze_intraday_volatility(open_price, high, low, close)
        
        # 4. Component Analysis (-1 to 1)
        component_signal = self._analyze_components(overnight_vol, open_close_vol, rogers_satchell_vol)
        
        # Combine signals
        composite_signal = (
            RegimeConfig.YangZhangConfig.WEIGHTS['trend'] * trend_signal +
            RegimeConfig.YangZhangConfig.WEIGHTS['overnight'] * overnight_signal +
            RegimeConfig.YangZhangConfig.WEIGHTS['intraday'] * intraday_signal +
            RegimeConfig.YangZhangConfig.WEIGHTS['component'] * component_signal
        )
        
        return max(min(composite_signal, 1.0), -1.0)
    
    def _calculate_overnight_volatility(self, close: pd.Series, open_price: pd.Series) -> pd.Series:
        """Calculate overnight (close-to-open) volatility."""
        overnight_returns = np.log(open_price / close.shift(1))
        return overnight_returns.rolling(window=RegimeConfig.YangZhangConfig.WINDOW).var()
    
    def _calculate_open_close_volatility(self, open_price: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate open-to-close volatility."""
        open_close_returns = np.log(close / open_price)
        return open_close_returns.rolling(window=RegimeConfig.YangZhangConfig.WINDOW).var()
    
    def _calculate_rogers_satchell_volatility(self, open_price: pd.Series, high: pd.Series, 
                                            low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Rogers-Satchell volatility."""
        rs_vol = (np.log(high / close) * np.log(high / open_price) + 
                 np.log(low / close) * np.log(low / open_price))
        return rs_vol.rolling(window=RegimeConfig.YangZhangConfig.WINDOW).mean()
    
    def _analyze_overnight_gaps(self, close: pd.Series, open_price: pd.Series) -> float:
        """Analyze overnight price gaps."""
        gaps = (open_price - close.shift(1)) / close.shift(1)
        recent_gaps = gaps.iloc[-RegimeConfig.YangZhangConfig.GAP_WINDOW:]
        gap_signal = recent_gaps.std() / gaps.std()
        return np.tanh((gap_signal - 1) * RegimeConfig.YangZhangConfig.GAP_FACTOR)
    
    def _analyze_intraday_volatility(self, open_price: pd.Series, high: pd.Series, 
                                   low: pd.Series, close: pd.Series) -> float:
        """Analyze intraday volatility patterns."""
        intraday_range = (high - low) / open_price
        range_sma = intraday_range.rolling(window=RegimeConfig.YangZhangConfig.INTRADAY_WINDOW).mean()
        range_signal = 0.0 if range_sma.iloc[-1] == 0 else (intraday_range.iloc[-1] - range_sma.iloc[-1]) / range_sma.iloc[-1]
        return np.tanh(range_signal * RegimeConfig.YangZhangConfig.INTRADAY_FACTOR)
    
    def _analyze_components(self, overnight_vol: pd.Series, open_close_vol: pd.Series, 
                          rogers_satchell_vol: pd.Series) -> float:
        """Analyze relationships between volatility components."""
        # Compare recent component ratios to historical averages
        overnight_ratio = 0.0 if overnight_vol.mean() == 0 else overnight_vol.iloc[-1] / overnight_vol.mean()
        open_close_ratio = 0.0 if open_close_vol.mean() == 0 else open_close_vol.iloc[-1] / open_close_vol.mean()
        rs_ratio = 0.0 if rogers_satchell_vol.mean() == 0 else rogers_satchell_vol.iloc[-1] / rogers_satchell_vol.mean()
        
        component_signal = (overnight_ratio + open_close_ratio + rs_ratio) / 3 - 1
        return np.tanh(component_signal * RegimeConfig.YangZhangConfig.COMPONENT_FACTOR)