import logging
import pandas as pd
from indicator_calculator import PrimaryTrendsSignalCalculator, MomentumSignalCalculator, VolatilitySignalCalculator, PatternSignalCalculator, TACalculator

logger = logging.getLogger(__name__)

class FeaturePreprocessor:
    def __init__(self):
        self.is_configured = False
        self.is_fitted = False

    def configure(self) -> None:
        self.is_configured = True

    def preprocess(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        output = {}
        for symbol, df in data.items():
            df = self._preprocess_symbol(df)
            output[symbol] = df
        return output

    def _preprocess_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame()
        output = TACalculator().calculate_indicators(df)
        return output

    def get_y(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        y = {}
        for symbol, df in data.items():
            rets = df['close'].pct_change()
            y[symbol] = rets
        return y