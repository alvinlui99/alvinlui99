# Standard library imports
import logging
import numpy as np
import pandas as pd

# Third-party imports


# Local application/library specific imports
from strategy import Strategy
from config import TradingConfig, ModelConfig

class LGBMstrategy(Strategy):
    """
    LightGBM-based trading strategy.
    
    Uses a trained LightGBM model to predict price movements and generate trading signals.
    """
    
    def __init__(self, model, threshold=None, logger=None):
        """
        Initialize the LightGBM strategy.
        
        Args:
            model: Trained LightGBM model
            threshold: Optional prediction threshold for generating signals (overrides config)
            logger: Optional logger instance
        """
        self.model = model
        self.threshold = threshold if threshold is not None else TradingConfig.SIGNAL_THRESHOLD
        self.position_size_pct = TradingConfig.MAX_ALLOCATION
        self.allow_shorts = TradingConfig.ALLOW_SHORTS
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Initialized LGBMstrategy with threshold: {self.threshold}")

    def get_signals(self, klines) -> dict:
        """
        Generate trading signals based on model predictions.
        
        Args:
            klines: Dictionary of pandas DataFrames containing price/volume data for each symbol
                   Format: {symbol: DataFrame}
                   
        Returns:
            dict: Dictionary of trading signals for each symbol
                  Format: {
                      symbol: {
                          "side": "BUY" or "SELL" or "NONE",
                          "type": "MARKET" or "LIMIT",
                          "quantity": float,
                          "price": float,
                          "confidence": float
                      }
                  }
        """
        # Initialize signals dictionary
        signals = {
            symbol: {
                "side": "NONE",
                "type": TradingConfig.DEFAULT_ORDER_TYPE,
                "quantity": 0.0,
                "price": 0.0,
                "confidence": 0.0
            }
            for symbol in klines.keys()
        }
        
        self.logger.info("Starting signal generation...")
        self.logger.info(f"Processing data for symbols: {list(klines.keys())}")
        
        try:
            # Verify we have valid data
            for symbol, df in klines.items():
                self.logger.info(f"Data for {symbol}:")
                self.logger.info(f"  Shape: {df.shape}")
                self.logger.info(f"  Columns: {df.columns.tolist()}")
                self.logger.info(f"  First timestamp: {df.index[0]}")
                self.logger.info(f"  Last timestamp: {df.index[-1]}")
                
                if df.empty:
                    self.logger.warning(f"Empty data for {symbol}")
                    continue
                    
                if 'Close' not in df.columns:
                    self.logger.warning(f"Missing Close price data for {symbol}")
                    continue
            
            # Get predictions from the model
            try:
                self.logger.info("Getting predictions from model...")
                predictions = self.model.predict(klines)
                self.logger.info(f"Got predictions for symbols: {list(predictions.keys())}")
                for symbol, pred in predictions.items():
                    self.logger.info(f"Prediction for {symbol}: shape={np.array(pred).shape}, values={pred[-5:] if len(pred) > 0 else 'empty'}")
            except Exception as e:
                self.logger.error(f"Error getting predictions: {str(e)}", exc_info=True)
                return signals
            
            # Process predictions for each symbol
            for symbol, pred in predictions.items():
                # Skip if we don't have predictions
                if not isinstance(pred, (list, np.ndarray)) or len(pred) == 0:
                    self.logger.warning(f"Invalid predictions for {symbol}: {pred}")
                    continue
                    
                # Get the last prediction (for the most recent candle)
                try:
                    latest_prediction = float(pred[-1])  # Ensure numeric
                    self.logger.info(f"{symbol} latest prediction: {latest_prediction:.6f}")
                except (IndexError, ValueError, TypeError) as e:
                    self.logger.error(f"Error processing prediction for {symbol}: {str(e)}")
                    continue
                
                # Get the current price from the klines data
                try:
                    current_price = float(klines[symbol]['Close'].iloc[-1])
                    self.logger.info(f"{symbol} current price: {current_price:.2f}")
                except (KeyError, IndexError, ValueError) as e:
                    self.logger.error(f"Could not get current price for {symbol}: {str(e)}")
                    continue
                
                # Skip if price is invalid
                if not current_price > 0:
                    self.logger.warning(f"Invalid price for {symbol}: {current_price}")
                    continue
                
                # Calculate predicted percent change
                predicted_pct_change = latest_prediction
                
                # Set signal confidence (absolute value of predicted change)
                signals[symbol]['confidence'] = abs(predicted_pct_change)
                
                # Determine trading action based on prediction
                try:
                    self.logger.info(f"{symbol} predicted change: {predicted_pct_change:.6f} (threshold: {self.threshold})")
                    if predicted_pct_change > self.threshold:
                        # Bullish signal - Buy
                        quantity = self._calculate_position_size(symbol, current_price)
                        if quantity > 0:
                            signals[symbol]['side'] = "BUY"
                            signals[symbol]['price'] = current_price
                            signals[symbol]['quantity'] = quantity
                            self.logger.info(f"Generated BUY signal for {symbol}: quantity={quantity:.3f}, price={current_price:.2f}")
                        
                    elif predicted_pct_change < -self.threshold and self.allow_shorts:
                        # Bearish signal - Sell (short)
                        quantity = self._calculate_position_size(symbol, current_price)
                        if quantity > 0:
                            signals[symbol]['side'] = "SELL"
                            signals[symbol]['price'] = current_price
                            signals[symbol]['quantity'] = quantity
                            self.logger.info(f"Generated SELL signal for {symbol}: quantity={quantity:.3f}, price={current_price:.2f}")
                    else:
                        self.logger.info(f"No signal generated for {symbol} (prediction within threshold)")
                    
                except Exception as e:
                    self.logger.error(f"Error calculating signal for {symbol}: {str(e)}")
                    continue
                
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}", exc_info=True)
        
        return signals
    
    def _calculate_position_size(self, symbol, price) -> float:
        """
        Calculate the position size based on available balance and risk parameters.
        This is a placeholder - implementation depends on your risk management approach.
        
        Args:
            symbol: The trading symbol
            price: Current price of the asset
            
        Returns:
            float: Quantity to trade
        """
        # Check if using fixed position size
        if TradingConfig.USE_FIXED_POSITION_SIZE:
            return TradingConfig.FIXED_POSITION_SIZE
        
        # This is a simplified example - in a real system, you would:
        # 1. Get account balance
        # 2. Apply position sizing rules based on your risk management
        # 3. Ensure the quantity meets minimum notional requirements
        
        # Calculate based on percentage of capital (to be replaced with actual balance in real system)
        assumed_capital = 1000.0  # Placeholder value
        allocated_capital = assumed_capital * self.position_size_pct
        position_size = allocated_capital / price
        
        # Round to appropriate precision (would be based on symbol info in real system)
        # Assuming 3 decimal places for most crypto
        position_size = round(position_size, 3)
        
        # Ensure minimum size
        min_size = 0.001  # Placeholder, would come from exchange info
        position_size = max(position_size, min_size)
        
        return position_size
    
    def adjust_parameters(self, threshold=None, position_size_pct=None, allow_shorts=None):
        """
        Update strategy parameters dynamically.
        
        Args:
            threshold: New threshold for signal generation
            position_size_pct: New position size percentage
            allow_shorts: Whether to allow short positions
        """
        if threshold is not None:
            self.threshold = threshold
            self.logger.info(f"Adjusted threshold to {threshold}")
        
        if position_size_pct is not None:
            self.position_size_pct = position_size_pct
            self.logger.info(f"Adjusted position size percentage to {position_size_pct}")
            
        if allow_shorts is not None:
            self.allow_shorts = allow_shorts
            self.logger.info(f"Set allow_shorts to {allow_shorts}")
            
    def get_strategy_info(self) -> dict:
        """
        Get information about the current strategy settings.
        
        Returns:
            Dictionary with strategy settings
        """
        return {
            "name": "LGBMstrategy",
            "threshold": self.threshold,
            "position_size_pct": self.position_size_pct,
            "allow_shorts": self.allow_shorts,
            "order_type": TradingConfig.DEFAULT_ORDER_TYPE
        }

