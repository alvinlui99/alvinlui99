library(keras)
library(tensorflow)
library(dplyr)
library(quantmod)

setwd("/Users/taryn.tsui/Documents/Alvin/GitHub/alvinlui99/R/Q Learning")

source("util_functions_v2.R")

csv_path <- "../Binance Data/1h/BTCUSDT.csv"
data <- read.csv(csv_path,
                 colClasses = c("index" = "POSIXct"))
timestep <- 20

model_input <- get_input_for_model(data = data,
                                   train_split = 0.8,
                                   timestep = timestep)


# Create training and test sets
# train_size <- floor(0.8 * nrow(data))
# train_data <- data[1:train_size, ]
# test_data <- data[(train_size + 1):nrow(data), ]


# Define LSTM model
model <- keras_model_sequential() %>%
  layer_lstm(units = 50,
             input_shape = c(timestep, 8),
             return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)

# Compile the model
model %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam"
)

# Reshape data for LSTM
# reshape_input <- function(data) {
#   array(data, dim = c(nrow(data),
#                       ncol(data),
#                       1))
# }

# x_train <- reshape_input(train_data)
# y_train <- train_data[, "Close"]

x_train <- model_input$x_train
y_train <- model_input$y_train

# Train the model
model %>% fit(x_train,
              y_train,
              epochs = 30,
              batch_size = 32,
              verbose = 1)

# Define Q-value approximator model
q_model <- keras_model_sequential() %>%
  layer_lstm(units = 50,
             input_shape = c(timestep, 8),
             return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 3)  # 3 actions: buy, sell, hold

# Compile the Q-value model
q_model %>% compile(
  optimizer = "adam",
  loss = "mse"
)

# Define parameters
actions <- c("buy", "sell", "hold")
learning_rate <- 0.01
discount_factor <- 0.95
epsilon <- 1  # Exploration factor
epsilon_min <- 0.01
epsilon_decay <- 0.995

test_data <- model_input$x_test
target_data <- model_input$y_test
q_values <- array_zeros <- array(0, dim = c(1, 3))

# Training the Q-approximator
for (t in 1:(nrow(test_data) - 1)) {
  current_state <- test_data[t, , , drop = FALSE]

  # Epsilon-greedy action selection
  if (runif(1) < epsilon) {
    action <- sample(actions, 1)
  } else {
    q_values <- predict(q_model, current_state, verbose = 0)
    action <- actions[which.max(q_values)]
  }

  # Calculate reward
  predicted_price <- predict(model, current_state, verbose = 0)
  actual_price <- target_data[t, ]
  reward <- ifelse(action == "buy" &&
                     actual_price > predicted_price, 1,
                   ifelse(action == "sell" &&
                            actual_price < predicted_price, 1, -1))

  # Get next state and max Q-value for next state
  next_state <- test_data[t + 1, , , drop = FALSE]
  future_q_values <- predict(q_model, next_state, verbose = 0)
  best_future_q <- max(future_q_values)

  # Update Q-value
  target_q_values <- q_values
  target_q_values[1, which(actions == action)] <-
    reward + discount_factor * best_future_q

  # Train Q-model
  q_model %>%
    fit(current_state,
        target_q_values,
        epochs = 1,
        verbose = 0)

  # Update exploration rate
  epsilon <- max(epsilon_min, epsilon * epsilon_decay)
}

# Summarize results
cat("Q-Learning with Q-value Approximator complete.\n")