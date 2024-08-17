# library(TTR)
# library(quantmod)
# library(dplyr)
# library(keras)
# library(tensorflow)
# library(ggplot2)

train_size <- 0.8
test_size  <- 0.2

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

model_input_x <- pca$x[1:(nrow(pca$x) - 1), 1:8]
model_input_y <- ohlcv_data_ind[(nrow(ohlcv_data_ind) -
                                   nrow(model_input_x) + 1):
                                  nrow(ohlcv_data_ind),
                                "ind_return"]

train_length <- as.integer(NROW(model_input_x) * train_size)

x_train <- model_input_x[1:train_length, ]
x_test <- model_input_x[as.integer(train_length + 1):NROW(model_input_x), ]

y_train <- model_input_y[1:train_length]
y_test <- model_input_y[as.integer(train_length + 1):NROW(model_input_x)]

# Reshape model_input_x for LSTM model (assuming a time series sequence)
# Reshape the data into [samples, time steps, features] for LSTM
x_train <- array(x_train,
                 dim = c(dim(x_train)[1], 1, dim(x_train)[2]))

y_train <- array(y_train,
                 dim = c(length(y_train), 1))

x_test <- array(x_test,
                dim = c(dim(x_test)[1], 1, dim(x_test)[2]))

y_test <- array(y_test,
                dim = c(length(y_test), 1))

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

# Train the LSTM model (Assuming you have target data model_input_y)
model %>% fit(
  x_train, y_train,
  validation_split = 0.25,
  epochs = 100,
  batch_size = 32
)

test_loss <- model %>% evaluate(x_test, y_test)
print("Test Loss:", test_loss)

# Make predictions using the model
predictions <- model %>% predict(x_test)

# Print predictions
print(predictions)

# Create a data frame with predictions and y_test
data <- data.frame(Predicted = predictions, Actual = y_test)

# Create a scatter plot using ggplot
ggplot(data, aes(x = Actual, y = Predicted)) +
  # Scatter plot points
  geom_point(color = "blue") +
  # Add a 45-degree reference line
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  # Labels and title
  labs(x = "Actual", y = "Predicted", title = "Actual vs Predicted Plot") +
  theme_minimal()  # Use a minimal theme