library(TTR)
library(quantmod)
library(dplyr)
library(keras)
library(tensorflow)

source("util_functions.R")

csv_folder <- "Binance Data/"
symbol <- "BTCUSDT"

csv_path <- paste0(csv_folder, symbol, ".csv")
ohlcv_data <- read.csv(csv_path, colClasses = c("index" = "POSIXct"))

ohlcv_data <- feature_generation(ohlcv_data)

ohlcv_data_ind <- ohlcv_data %>%
  select(starts_with("ind_"))

pca <- ohlcv_data_ind %>%
  na.omit() %>%
  scale() %>%
  prcomp()

x_train <- pca$rotation[, 1:8]


# Reshape x_train for LSTM model (assuming a time series sequence)
# Reshape the data into [samples, time steps, features] for LSTM
x_train <- array(x_train,
                 dim = c(dim(x_train)[1], 1, dim(x_train)[2]))

# Build the LSTM model
model <- keras_model_sequential()
model %>%
  layer_lstm(units = 50, input_shape = c(1, 8)) %>%
  layer_dense(units = 1)  # Example output layer, adjust as needed

# Compile the model
model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam()
)

# Train the LSTM model (Assuming you have target data y_train)
model %>% fit(
  x_train, y_train,
  epochs = 100,
  batch_size = 32
)

# Make predictions using the model
predictions <- model %>% predict(x_train)

# Print predictions
print(predictions)

length(x_train)
length(predictions)