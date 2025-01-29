from typing import Dict

class LeverageStrategy:
    def __init__(self):
        self.is_configured = False

    def configure(self):
        self.is_configured = True

    def get_leverages(self) -> Dict[str, int]:
        pass