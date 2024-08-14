library(TTR)
library(quantmod)
library(dplyr)
library(keras)
library(tensorflow)

source("util_functions.R")

csv_folder <- 'Binance Data/'
symbol = 'BTCUSDT'

csv_path <- paste0(csv_folder, symbol, '.csv')
OHLCV <- read.csv(csv_path, colClasses = c("index" = "POSIXct"))

OHLCV <- feature_generation(OHLCV)

OHLCV_ind <- OHLCV %>%
  select(starts_with("ind_"))

pca <- OHLCV_ind %>%
  na.omit() %>%
  scale() %>%
  prcomp()

input_X <- pca$rotation[, 1:8]


# Reshape input_X for LSTM model (assuming a time series sequence)
# Reshape the data into [samples, time steps, features] for LSTM
input_X <- array(input_X, dim = c(dim(input_X)[1], 1, dim(input_X)[2]))

# Build the LSTM model
model <- keras_model_sequential()
model %>%
  layer_lstm(units = 50, input_shape = c(1, 8)) %>%
  layer_dense(units = 1)  # Example output layer, adjust as needed

# Compile the model
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam()
)

# Train the LSTM model (Assuming you have target data y_train)
model %>% fit(
  input_X, y_train,
  epochs = 100,
  batch_size = 32
)

# Make predictions using the model
predictions <- model %>% predict(input_X)

# Print predictions
print(predictions)




