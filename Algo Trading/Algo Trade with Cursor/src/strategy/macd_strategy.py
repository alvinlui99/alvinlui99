from typing import Dict
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from .indicators import calculate_macd, calculate_rsi

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
                    'take_profit': None
                }
                continue
                
            prev = symbol_data.iloc[-2]
            
            # Initialize signal
            signal = {
                'action': 'HOLD',
                'strength': 0,
                'price': latest['Close'],
                'stop_loss': None,
                'take_profit': None
            }
            
            # MACD crossover signals
            if (prev['macd'] < prev['signal'] and 
                latest['macd'] > latest['signal'] and 
                latest['rsi'] < self.rsi_oversold):
                signal['action'] = 'BUY'
                signal['strength'] = 1
                signal['stop_loss'] = latest['Close'] * 0.98  # 2% stop loss
                signal['take_profit'] = latest['Close'] * 1.04  # 4% take profit
                
            elif (prev['macd'] > prev['signal'] and 
                  latest['macd'] < latest['signal'] and 
                  latest['rsi'] > self.rsi_overbought):
                signal['action'] = 'SELL'
                signal['strength'] = 1
                signal['stop_loss'] = latest['Close'] * 1.02  # 2% stop loss
                signal['take_profit'] = latest['Close'] * 0.96  # 4% take profit
            
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
        
        # Calculate position size based on risk
        position_size = risk_amount / stop_loss_distance
        
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
            return False
            
        # Check if take profit is at least 2x the stop loss distance
        stop_loss_distance = abs(signal['price'] - signal['stop_loss'])
        take_profit_distance = abs(signal['price'] - signal['take_profit'])
        
        return take_profit_distance >= (2 * stop_loss_distance) 