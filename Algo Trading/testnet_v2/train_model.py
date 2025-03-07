from config import setup_logging
from config import ModelConfig
from utils import utils
from models import LGBMRegressorModel

if __name__ == "__main__":
    setup_logging()
    data = utils.read_klines_from_csv(
        'data', '1h',
        start_date=utils.convert_str_to_datetime(ModelConfig.TRAIN_START_DATE),
        end_date=utils.convert_str_to_datetime(ModelConfig.TRAIN_END_DATE))
    model = LGBMRegressorModel()
    model.train(data)
