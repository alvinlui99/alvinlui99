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
        self.trading_strategy = None

    def run(self):
        """Main trading cycle that runs every hour"""
        try:
            self._setup_trading_cycle()

            logging.info(f"Starting trading cycle at {datetime.now()}")
            
            market_data = self._get_current_market_data()

            params = {
                'data': market_data
            }
            signals = self.trading_strategy.get_signals(**params)

            if signals:
                self._execute_trades(signals)

            logging.info(f"Completed trading cycle at {datetime.now()}")
        except Exception as e:
            logging.error(f"Error in trading cycle: {str(e)}")
            logging.error("Full error:", exc_info=True)
    
    def _setup_trading_cycle(self):
        if self.trading_strategy is None:
            preprocessor = self._prepare_preprocessor()
            model = self._prepare_model()
            leverage_strategy = self._prepare_leverage_strategy()
            self._prepare_trading_strategy(
                preprocessor=preprocessor,
                model=model,
                leverage_strategy=leverage_strategy
            )

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

    def get_historical_klines(self) -> dict[str, pd.DataFrame]:
        data = {}
        start_time = utils.convert_str_to_datetime(ModelConfig.TRAIN_START_DATE)
        end_time = utils.convert_str_to_datetime(ModelConfig.TRAIN_END_DATE)
        for symbol in TradingConfig.SYMBOLS:
            df = self.binance_service.get_historical_klines(symbol,
                                                            start_time,
                                                            end_time,
                                                            TradingConfig.TIMEFRAME)
            data[symbol] = df
        return data

    def _get_current_market_data(self) -> dict[str, pd.DataFrame]:
        return self.binance_service.get_current_klines_multi(TradingConfig.SYMBOLS, TradingConfig.TIMEFRAME)

    def _execute_trades(self, signals: List[Dict[str, float]]):
        pass
