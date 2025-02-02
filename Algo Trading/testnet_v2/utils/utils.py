import datetime
import pandas as pd

def convert_str_to_datetime(date_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(date_str, '%Y-%m-%d').replace(hour=0, minute=0, second=0, tzinfo=datetime.timezone.utc)

def save_historical_klines(data: dict[str, pd.DataFrame], path: str):
    for symbol, df in data.items():
        df.to_csv(f"{path}/{symbol}.csv", index=False)
