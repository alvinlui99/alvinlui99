# Q Learning Agent should behave like a human trader
# It should learn and then make dsecision based on info
#    received at time t
# It should forecast or predict, but not receive
#    future info

# Data Structure
# State: (1, timestep, features) for LSTM model
# target: numeric
# target_f: (1, action_size)
# 

library(keras)

build_q_network <- function(input_shape, action_size) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 64,
               input_shape = input_shape,
               return_sequences = TRUE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_lstm(units = 64) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 32,
                activation = "relu") %>%
    layer_dense(units = action_size,
                activation = "linear")

  model %>% compile(
    optimizer = "adam",
    loss = "mse"
  )

  return(model)
}

QLearningAgent <- function(timestep, action_size) {
  agent <- list(
    timestep = timestep,
    action_size = action_size,
    model = build_q_network(c(timestep, 8), action_size),
    gamma = 0.95,
    epsilon = 1.0,
    epsilon_decay = 0.995,
    epsilon_min = 0.01
  )

  agent$act <- function(state) {
    if (runif(1) <= agent$epsilon) {
      return(sample(0:(agent$action_size - 1), 1))  # Explore
    }
    q_values <- predict(agent$model, state)
    return(which.max(q_values) - 1)  # Exploit
  }

  agent$train <- function(state, action, reward, next_state, done) {
    target <- reward
    if (!done) {
      target <- target + agent$gamma * max(predict(agent$model, next_state))
    }
    target_f <- predict(agent$model, state)
    target_f[1, action + 1] <- target
    fit(agent$model, state, target_f, epochs = 1, verbose = 0)
    if (agent$epsilon > agent$epsilon_min) {
      agent$epsilon <- agent$epsilon * agent$epsilon_decay
    }
  }

  return(agent)
}