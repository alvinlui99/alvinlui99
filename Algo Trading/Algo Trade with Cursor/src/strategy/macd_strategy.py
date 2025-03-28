from typing import Dict
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from .indicators import calculate_macd, calculate_rsi
import logging
import math

logger = logging.getLogger(__name__)

class MACDStrategy(BaseStrategy):
    def __init__(self, 
                 trading_pairs: list,
                 timeframe: str = '1h',
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 risk_per_trade: float = 0.02):
        """
        Initialize MACD Strategy.
        
        Args:
            trading_pairs (list): List of trading pairs
            timeframe (str): Timeframe for analysis
            rsi_period (int): Period for RSI calculation
            rsi_overbought (float): RSI overbought threshold
            rsi_oversold (float): RSI oversold threshold
            risk_per_trade (float): Risk per trade as percentage of account
        """
        super().__init__(trading_pairs, timeframe)
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.risk_per_trade = risk_per_trade
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD and RSI indicators.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with indicators
        """
        # Make sure we have a copy to avoid modifying the original
        data = data.copy()
        
        # Ensure the necessary columns exist with proper case
        if 'Close' not in data.columns:
            if 'close' in data.columns:
                data['Close'] = data['close']
            elif 'closePrice' in data.columns:
                data['Close'] = data['closePrice']
        
        # Calculate MACD
        macd_line, signal_line, histogram = calculate_macd(data, column='Close')
        data['macd'] = macd_line
        data['signal'] = signal_line
        data['histogram'] = histogram
        
        # Calculate RSI
        data['rsi'] = calculate_rsi(data, self.rsi_period, column='Close')
        
        return data
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate trading signals based on MACD and RSI.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary of price data with symbol as key
            
        Returns:
            Dict[str, Dict]: Trading signals
        """
        signals = {}
        
        for symbol in self.trading_pairs:
            if symbol not in data:
                continue
                
            # Get the DataFrame for this symbol
            symbol_data = data[symbol]
            
            # Calculate indicators if not already done
            if 'macd' not in symbol_data.columns:
                symbol_data = self.calculate_indicators(symbol_data)
            
            # Get latest values
            latest = symbol_data.iloc[-1]
            
            # Make sure we have at least 2 rows
            if len(symbol_data) < 2:
                signals[symbol] = {
                    'action': 'HOLD',
                    'strength': 0,
                    'price': latest['Close'],
                    'stop_loss': None,
                    'take_profit': None,
                    'reason': 'Not enough data'
                }
                continue
                
            prev = symbol_data.iloc[-2]
            
            # Initialize signal
            signal = {
                'action': 'HOLD',
                'strength': 0,
                'price': latest['Close'],
                'stop_loss': None,
                'take_profit': None,
                'reason': ''
            }
            
            # Debug values
            macd_above_signal = latest['macd'] > latest['signal']
            macd_rising = latest['macd'] > prev['macd']
            macd_crossover = (prev['macd'] < prev['signal'] and latest['macd'] > latest['signal'])
            macd_crossunder = (prev['macd'] > prev['signal'] and latest['macd'] < latest['signal'])
            rsi_oversold = latest['rsi'] < self.rsi_oversold
            rsi_overbought = latest['rsi'] > self.rsi_overbought
            
            # Generate signals with more relaxed conditions
            # Buy conditions:
            # 1. MACD crossover above signal line OR
            # 2. MACD above signal line and rising OR
            # 3. RSI oversold with MACD above signal line
            if ((macd_crossover or (macd_above_signal and macd_rising) or (rsi_oversold and macd_above_signal))):
                signal['action'] = 'BUY'
                signal['strength'] = 1
                signal['stop_loss'] = latest['Close'] * 0.98  # 2% stop loss
                signal['take_profit'] = latest['Close'] * 1.04  # 4% take profit
                signal['reason'] = 'MACD bullish pattern or oversold RSI with MACD above signal'
            
            # Sell conditions:
            # 1. MACD crossover below signal line OR
            # 2. MACD below signal line and falling OR
            # 3. RSI overbought with MACD below signal line
            elif ((macd_crossunder or (not macd_above_signal and not macd_rising) or (rsi_overbought and not macd_above_signal))):
                signal['action'] = 'SELL'
                signal['strength'] = 1
                signal['stop_loss'] = latest['Close'] * 1.02  # 2% stop loss
                signal['take_profit'] = latest['Close'] * 0.96  # 4% take profit
                signal['reason'] = 'MACD bearish pattern or overbought RSI with MACD below signal'
            else:
                # Provide reasons why no trade signal was generated
                reasons = []
                if not macd_crossover and not (macd_above_signal and macd_rising) and not (rsi_oversold and macd_above_signal):
                    reasons.append(f"No bullish pattern (crossover: {macd_crossover}, above signal: {macd_above_signal}, rising: {macd_rising}, RSI oversold: {rsi_oversold})")
                if not macd_crossunder and not (not macd_above_signal and not macd_rising) and not (rsi_overbought and not macd_above_signal):
                    reasons.append(f"No bearish pattern (crossunder: {macd_crossunder}, below signal: {not macd_above_signal}, falling: {not macd_rising}, RSI overbought: {rsi_overbought})")
                
                signal['reason'] = '; '.join(reasons)
            
            signals[symbol] = signal
        
        self.signals = signals
        return signals
    
    def calculate_position_size(self, 
                              symbol: str, 
                              signal: Dict,
                              account_balance: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol (str): Trading pair
            signal (Dict): Trading signal
            account_balance (float): Account balance
            
        Returns:
            float: Position size
        """
        if signal['action'] == 'HOLD':
            return 0
            
        risk_amount = account_balance * self.risk_per_trade
        stop_loss_distance = abs(signal['price'] - signal['stop_loss'])
        
        # Calculate initial position size based on risk
        position_size = risk_amount / stop_loss_distance
        
        # Calculate maximum affordable position size considering commission
        max_position = account_balance / (signal['price'] * (1 + 0.0004))  # 0.0004 is commission
        
        # Take the minimum of risk-based size and affordable size
        position_size = min(position_size, max_position)
        
        # Round down to 3 decimal places to ensure we can afford it
        position_size = math.floor(position_size * 1000) / 1000
        
        # logger.info(f"Calculating position size for {signal['action']}:")
        # logger.info(f"  Account Balance: {account_balance:.2f} USDT")
        # logger.info(f"  Risk Amount ({self.risk_per_trade*100}%): {risk_amount:.2f} USDT")
        # logger.info(f"  Price: {signal['price']:.2f} USDT")
        # logger.info(f"  Stop Loss Distance: {stop_loss_distance:.2f} USDT")
        # logger.info(f"  Max Affordable Size: {max_position:.6f} BTC")
        # logger.info(f"  Risk-Based Size: {risk_amount / stop_loss_distance:.6f} BTC")
        # logger.info(f"  Final Position Size: {position_size:.6f} BTC")
        # logger.info(f"  Position Value: {position_size * signal['price']:.2f} USDT")
        # logger.info(f"  Total Cost with Commission: {position_size * signal['price'] * 1.0004:.2f} USDT")
        
        # Ensure minimum position size (0.001 BTC for Binance)
        if position_size < 0.001:
            logger.info(f"Position size {position_size:.6f} BTC is below minimum (0.001 BTC)")
            return 0
            
        return position_size
    
    def validate_signal(self, signal: Dict) -> bool:
        """
        Validate if a signal meets the strategy's criteria.
        
        Args:
            signal (Dict): Trading signal
            
        Returns:
            bool: True if signal is valid
        """
        if signal['action'] == 'HOLD':
            return True
            
        # Check if stop loss and take profit are set
        if not signal['stop_loss'] or not signal['take_profit']:
            logger.info(f"Signal rejected: Missing stop loss or take profit - Stop Loss: {signal['stop_loss']}, Take Profit: {signal['take_profit']}")
            return False
            
        # Calculate distances
        stop_loss_distance = abs(signal['price'] - signal['stop_loss'])
        take_profit_distance = abs(signal['price'] - signal['take_profit'])
        ratio = take_profit_distance / stop_loss_distance if stop_loss_distance != 0 else 0
        
        # logger.info(f"Validating {signal['action']} signal:")
        # logger.info(f"  Price: {signal['price']:.2f}")
        # logger.info(f"  Stop Loss: {signal['stop_loss']:.2f} (Distance: {stop_loss_distance:.2f})")
        # logger.info(f"  Take Profit: {signal['take_profit']:.2f} (Distance: {take_profit_distance:.2f})")
        # logger.info(f"  Ratio: {ratio:.2f}x (Required: >= 1.5x)")
        
        is_valid = take_profit_distance >= (1.5 * stop_loss_distance)
        if not is_valid:
            logger.info("Signal rejected: Take profit distance not >= 1.5x stop loss distance")
        
        return is_valid 