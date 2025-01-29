import datetime

def convert_str_to_datetime(date_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(date_str, '%Y-%m-%d').replace(hour=0, minute=0, second=0, tzinfo=datetime.timezone.utc)