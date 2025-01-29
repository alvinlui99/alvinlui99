from services import BinanceService
import datetime

def set_time(
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0
    ) -> datetime.datetime:
    return datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc)

if __name__ == '__main__':
    binance_service = BinanceService()
    start_time = set_time(2024, 1, 1)
    end_time = set_time(2024, 1, 2)
    df = binance_service.get_historical_data('BTCUSDT', start_time, end_time, '1h')
    print(df)