# Standard library imports
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

# Third-party imports

# Local application/library specific imports
from config import BaseConfig, setup_logging
from utils import utils
from model.LGBMmodel import LGBMmodel

def setup_custom_logging():
    """Set up custom logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler('logs/train_model.log')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def main():
    """Main function to train the model and evaluate predictions."""
    # Set up logging
    logger = setup_custom_logging()
    logger.info("Starting model training process")
    
    try:
        # Load and split data
        logger.info(f"Loading data for symbols: {BaseConfig.SYMBOLS}")
        dfs = utils.load_dfs_from_csv(BaseConfig.SYMBOLS, 'data/klines.csv')
        logger.info(f"Loaded data shapes: {[(symbol, df.shape) for symbol, df in dfs.items()]}")
        
        logger.info("Splitting data into train/val/test sets")
        train_dfs, val_dfs, test_dfs = utils.split_dfs(dfs)
        logger.info(f"Train data rows: {[len(df) for df in train_dfs.values()]}")
        logger.info(f"Val data rows: {[len(df) for df in val_dfs.values()]}")
        logger.info(f"Test data rows: {[len(df) for df in test_dfs.values()]}")
        
        # Create and train model
        logger.info("Initializing model")
        model = LGBMmodel(BaseConfig.SYMBOLS, logger=logger)
        
        logger.info("Training model")
        model.train(train_dfs, val_dfs)
        
        # Save model
        os.makedirs('model/trained_models', exist_ok=True)
        logger.info("Saving trained model")
        model.save_model('model/trained_models')
        
        # Generate predictions
        logger.info("Generating predictions on test data")
        predictions = model.predict(test_dfs)
        
        # Evaluate results
        logger.info("Evaluating model performance")
        for symbol in BaseConfig.SYMBOLS:
            actual = test_dfs[symbol]['Close'].pct_change(fill_method=None).shift(-1).values[:-1]
            predicted = predictions[symbol]
            
            # Check if predictions are available
            if len(predicted) > 0:
                # Calculate RMSE
                min_len = min(len(actual), len(predicted))
                if min_len > 0:
                    actual = actual[:min_len]
                    predicted = predicted[:min_len]
                    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                    logger.info(f"RMSE for {symbol}: {rmse}")
                    
                    # Plot predictions vs actual
                    plt.figure(figsize=(12, 6))
                    plt.plot(actual, label='Actual')
                    plt.plot(predicted, label='Predicted')
                    plt.title(f'Actual vs Predicted Returns for {symbol}')
                    plt.legend()
                    os.makedirs('plots', exist_ok=True)
                    plt.savefig(f'plots/{symbol}_predictions.png')
                    plt.close()
                else:
                    logger.warning(f"Not enough data points to calculate RMSE for {symbol}")
            else:
                logger.warning(f"No predictions available for {symbol}")
                
        logger.info("Model training and evaluation completed")
        
    except Exception as e:
        logger.exception(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    main()