from typing import List, Dict, Optional, TypedDict
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from services import BinanceService, PortfolioService, TradingService
from models import Model, LGBMRegressorModel, FeaturePreprocessor
from strategy import MLStrategy, RegimeLeverageStrategy, TradingStrategy, LeverageStrategy
from config import BaseConfig, TradingConfig, RiskConfig, FeatureConfig, ModelConfig
from utils import utils

class TradingCycle:
    def __init__(self):
        self.binance_service = BinanceService()
        self.portfolio_service = PortfolioService(self.binance_service.client)
        self.trading_service = TradingService(self.binance_service.client)

        self.preprocessor = None
        self.model = None
        self.leverage_strategy = None
        self.trading_strategy = None

    def run(self):
        """Main trading cycle that runs every hour"""
        try:
            self._setup_trading_cycle()

            logging.info(f"Starting trading cycle at {datetime.now()}")
            
            market_data = self.get_current_market_data()

            if self._check_stop_loss():
                leverages = {symbol: 1 for symbol in TradingConfig.SYMBOLS}
                signals = self.trading_strategy.get_signals(
                    market_data,
                    self.get_current_weights(),
                    self.get_current_prices(),
                    self.get_account_value(),
                    leverages,
                    stop_loss_active=True
                )
            else:
                leverages = self._get_leverages(market_data)
                signals = self.trading_strategy.get_signals(
                    self.preprocessor.get_transformed_input(market_data),
                    self.get_current_weights(),
                    self.get_current_prices(),
                    self.get_account_value(),
                    leverages,
                    stop_loss_active=False
                )

            if signals:
                self.trading_service.set_all_leverages(leverages)
                trade_instructions = self._generate_trade_instructions(signals, self.get_current_weights())
                logging.info(f"Setting leverages: {leverages}")
                logging.info(f"Setting signals: {signals}")
                # self._execute_trades(trade_instructions)
                logging.info(f"Executed trades at {datetime.now()}")
            
            logging.info(f"Completed trading cycle at {datetime.now()}")
        except Exception as e:
            logging.error(f"Error in trading cycle: {str(e)}")
            logging.error("Full error:", exc_info=True)

    def get_market_data(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        interval: str = '1h'
    ) -> Dict[str, pd.DataFrame]:
        """Fetch and process market data"""
        try:    
            symbol_data = self.binance_service.get_historical_data_multi(
                symbols=TradingConfig.SYMBOLS,
                start_time=start_time,
                end_time=end_time,
                interval=interval
            )
            for symbol, df in symbol_data.items():
                if df.empty:
                    logging.error(f"Empty DataFrame received for {symbol}")
                if 'close' not in df.columns:
                    logging.error(f"Missing 'close' column for {symbol}. Available columns: {df.columns.tolist()}")
            return symbol_data
        except Exception as e:
            logging.error(f"Error fetching market data: {str(e)}")
            raise

    def get_current_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch and process current market data"""
        start_time=datetime.now() - timedelta(days=50)
        end_time=datetime.now()
        return self.get_market_data(start_time, end_time)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights"""
        return self.portfolio_service.get_current_weights()

    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        return self.binance_service.get_current_price_multi(TradingConfig.SYMBOLS)

    def get_account_value(self) -> float:
        """Get current account value"""
        return self.portfolio_service.get_account_value()

    def _get_leverages(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        leverages = {}
        for symbol, data in market_data.items():
            leverages[symbol] = self.leverage_strategy.get_leverages(data)
        return leverages

    def prepare_raw_training_dataset(self) -> Dict[str, pd.DataFrame]:
        train_start_time = utils.convert_str_to_datetime(ModelConfig.TRAIN_START_DATE)
        train_end_time = utils.convert_str_to_datetime(ModelConfig.TRAIN_END_DATE)
        val_start_time = utils.convert_str_to_datetime(ModelConfig.VAL_START_DATE)
        val_end_time = utils.convert_str_to_datetime(ModelConfig.VAL_END_DATE)

        if BaseConfig.LOAD_TRAINING_DATA_SWITCH:
            train_data_X = self.preprocessor.load_training_data(train_start_time, train_end_time)
            val_data_X = self.preprocessor.load_training_data(val_start_time, val_end_time)
        else:
            train_data_X = self.get_market_data(train_start_time, train_end_time)
            val_data_X = self.get_market_data(val_start_time, val_end_time)

        train_data_y = self.preprocessor.get_returns(train_data_X)
        val_data_y = self.preprocessor.get_returns(val_data_X)
        dataset = {
            'train_data_X': train_data_X,
            'train_data_y': train_data_y,
            'val_data_X': val_data_X,
            'val_data_y': val_data_y
        }

        return dataset

    def _prepare_preprocessor(self) -> FeaturePreprocessor:
        preprocessor = FeaturePreprocessor()
        preprocessor.configure()
        if BaseConfig.LOAD_SCALER_SWITCH:
            preprocessor.load_scaler()
        else:
            dataset_raw = self.prepare_raw_training_dataset()
            dataset = self.preprocessor.add_features(dataset_raw)
            preprocessor.fit(dataset)
        return preprocessor

    def _prepare_model(self) -> Model:
        model = LGBMRegressorModel()
        if BaseConfig.LOAD_MODEL_SWITCH:
            model.load_model()
        else:
            dataset = self.prepare_training_dataset()
            train_data_X_with_features = self.preprocessor.prepare_features_for_symbols(dataset['train_data_X'])
            val_data_X_with_features = self.preprocessor.prepare_features_for_symbols(dataset['val_data_X'])
            train_data_X_transformed = self.preprocessor.transform(train_data_X_with_features)
            val_data_X_transformed = self.preprocessor.transform(val_data_X_with_features)
            dataset = {
                'train_data_X': train_data_X_transformed,
                'train_data_y': dataset['train_data_y'],
                'val_data_X': val_data_X_transformed,
                'val_data_y': dataset['val_data_y']
            }
            model.train(dataset)
        return model

    def _prepare_leverage_strategy(self) -> LeverageStrategy:
        leverage_strategy = RegimeLeverageStrategy()
        leverage_strategy.configure()
        return leverage_strategy

    def _prepare_trading_strategy(
        self,
        preprocessor: FeaturePreprocessor,
        model: Model,
        leverage_strategy: LeverageStrategy
    ) -> TradingStrategy:
        self.trading_strategy = MLStrategy()
        self.trading_strategy.configure(
            features=FeatureConfig.FEATURE_NAMES,
            symbols=TradingConfig.SYMBOLS,
            preprocessor=preprocessor,
            model=model,
            leverage_strategy=leverage_strategy
        )

    def _check_stop_loss(self) -> bool:
        """Check if stop loss is active"""
        pnl = self.portfolio_service.get_pnl()
        account_value = self.get_account_value()
        if pnl / account_value < RiskConfig.STOP_LOSS_THRESHOLD:
            return True
        return False
    
    def _execute_trades(self, instructions: Dict[str, Dict[str, float]]):
        """Execute trades based on generated signals"""
        for symbol, instruction in instructions.items():
            self.trading_service.place_order(
                symbol=symbol,
                side=instruction['action'],
                quantity=instruction['size']
            )

    def _generate_trade_instructions(self, signals: Dict[str, float], current_sizes: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Generate trade instructions based on generated signals"""
        instructions = {}
        for symbol, signal in signals:
            target_size = signal
            current_size = current_sizes[symbol]
            trade_size = target_size - current_size
            if trade_size / current_size > TradingConfig.MIN_POSITION_CHANGE:
                if trade_size > 0:
                    instructions[symbol] = {'action': 'buy', 'size': trade_size}
                elif trade_size < 0:
                    instructions[symbol] = {'action': 'sell', 'size': -trade_size}
        return instructions
    
    def _setup_trading_cycle(self):
        if self.trading_strategy is None:
            preprocessor = self._prepare_preprocessor()
            model = self._prepare_model
            leverage_strategy = self._prepare_leverage_strategy
            self._prepare_trading_strategy(
                preprocessor=preprocessor,
                model=model,
                leverage_strategy=leverage_strategy
            )