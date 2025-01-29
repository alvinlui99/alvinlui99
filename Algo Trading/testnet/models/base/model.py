class Model:
    def __init__(self):
        self.is_configured = False
        self.is_trained = False

    def train(self):
        if not self.is_configured:
            raise ValueError("Model must be configured before training")

    def predict_weights(self):
        """Make predictions using the loaded model"""
        if not self.is_trained or not self.is_configured:
            raise ValueError("Model must be trained and configured before prediction")