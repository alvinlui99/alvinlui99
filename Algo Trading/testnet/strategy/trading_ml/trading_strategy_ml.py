from typing import List, Dict
import numpy as np
from models import Model
from .. import TradingStrategy, LeverageStrategy
from config import TradingConfig, FeatureConfig

class MLStrategy(TradingStrategy):
    def __init__(self):
        super().__init__()

    def configure(
        self,
        features: List[str] = FeatureConfig.FEATURE_NAMES,
        symbols: List[str] = TradingConfig.SYMBOLS,
        model: Model = None,
        leverage_strategy: LeverageStrategy = None
    ) -> None:
        self.features = features
        self.symbols = symbols
        self.model = model
        self.leverage_strategy = leverage_strategy
        self.is_configured = True
        
    def get_signals(
        self,
        data: np.ndarray,
        current_weights: Dict[str, float],
        current_prices: Dict[str, float],
        budget: float,
        leverages: Dict[str, int],
        stop_loss_active: bool = False
    ) -> Dict[str, float]:
        if not self.is_configured:
            raise ValueError("Strategy must be configured before use")
        
        target_weights = self.model.predict_weights(data)
        
        if not self.should_rebalance(current_weights, target_weights, stop_loss_active):
            return {}
        
        filtered_weights = self.filter_trades(current_weights, target_weights)
        if filtered_weights == current_weights:
            return {}

        signals = {}
        for symbol, weight in filtered_weights.items():
            price = current_prices[symbol]
            signals[symbol] = (weight * budget * leverages[symbol]) / price
        return signals

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        stop_loss_active: bool
    ) -> bool:
        # If no current weights, rebalance
        if not current_weights:
            return True
        
        total_deviation = sum(abs(target_weights.get(symbol, 0) - current_weights.get(symbol, 0))
                                for symbol in self.symbols)

        if stop_loss_active:
            return total_deviation > TradingConfig.REBALANCE_THRESHOLD * TradingConfig.STOP_LOSS_REBALANCE_THRESHOLD_MULTIPLIER
        else:
            return total_deviation > TradingConfig.REBALANCE_THRESHOLD

    def filter_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        filtered_weights = {symbol: 0 for symbol in self.symbols}
        to_be_adjusted_weights = {}

        for symbol in self.symbols:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            weight_change = abs(target_weight - current_weight)

            if weight_change < TradingConfig.REBALANCE_THRESHOLD:
                filtered_weights[symbol] = current_weight
            else:
                to_be_adjusted_weights[symbol] = target_weight
        
        total_adjusted_weight = sum(to_be_adjusted_weights.values())
        total_filtered_weight = sum(filtered_weights.values())
        if total_adjusted_weight > 0:
            filtered_weights = {
                symbol: weight / total_adjusted_weight * (1 - total_filtered_weight) 
                for symbol, weight in to_be_adjusted_weights.items()
            }

        # Check if filtered weights sum up to 1
        if abs(sum(filtered_weights.values()) - 1) > 1e-3:
            raise ValueError("Filtered weights do not sum up to 1")
        
        return filtered_weights
