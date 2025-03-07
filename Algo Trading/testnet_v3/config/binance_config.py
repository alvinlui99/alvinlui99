import os
from dotenv import load_dotenv
from typing import Optional
load_dotenv()

class BinanceConfig:
    @staticmethod
    def get_env_variable(var_name: str) -> Optional[str]:
        value = os.getenv(var_name)
        if not value:
            raise ValueError(f"Environment variable {var_name} is not set")
        return value
    API_KEY = get_env_variable('API_KEY')
    API_SECRET = get_env_variable('API_SECRET')
    BASE_URL = get_env_variable('URL_TESTNET')
