library(keras)
library(tensorflow)
library(dplyr)
library(quantmod)

setwd("/Users/taryn.tsui/Documents/Alvin/GitHub/alvinlui99/R")

source("util_functions_QLearning.R")
source("trading_env_QLearning.R")
source("Q_Agent.R")

trade <- function(agent, environment, episodes) {
  for (e in 1:episodes) {
    environment$reset()
    state <- environment$get_state()
    done <- FALSE
    total_profit <- 0

    while (!done) {
      action <- agent$act(state)
      result <- environment$step(action)
      reward <- result$reward
      done <- result$done

      next_state <- environment$get_state()
      agent$train(state, action, reward, next_state, done)
      state <- next_state
      total_profit <- total_profit + reward

      print(environment$current_index)

      # Determine trade size using Kelly Criterion
      # q_values <- predict(agent$model, state)
      # trade_size <- kelly_criterion(q_values)
    }

    cat(sprintf("Episode %d/%d - Total Profit: %f\n",
                e, episodes, total_profit))
  }
}

# Example usage
csv_path <- "Binance Data/1h/BTCUSDT.csv"
data <- read.csv(csv_path,
                 colClasses = c("index" = "POSIXct"))
timestep <- 20

model_input <- get_input_for_model(data = data,
                                   train_split = 0.1,
                                   timestep = timestep)

x_train <- model_input$x_train
y_train <- model_input$y_train
price_data <- model_input$close_train
environment <- TradingEnvironment(price_data, indicators = x_train)

agent <- QLearningAgent(timestep = timestep, action_size = 3)  # Example sizes
trade(agent, environment, episodes = 2)
