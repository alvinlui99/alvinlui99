library(quantmod)
library(rlang)

interval <- "1h/"

csv_folder <- "Binance Data/"
files_list <- list.files(paste0(csv_folder, interval), full.names = TRUE)
files_list
symbols <- sub(".csv", "", files_list)

symbol <- "BTCUSDT"

# Load stock data
csv_path <- paste0(csv_folder, interval, symbol, ".csv")
data <- read.csv(csv_path,
                 colClasses = c("index" = "POSIXct"))

# Define a simple Q-learning agent
Q_learning_agent <- function(data,
                             epsilon = 0.1,
                             alpha = 0.1,
                             gamma = 0.9,
                             n_episodes = 100) {
  Q <- matrix(0, nrow = 2, ncol = 2)  # Q-table with 2 actions and 2 states
  state <- 1  # Initial state
  profit <- 0  # Initial profit

  for (episode in 1:n_episodes) {
    action <- sample(1:2, 1)  # Random action selection
    new_state <- sample(1:2, 1)  # Random state transition

    # Simulate profit based on action and state
    reward <- data[new_state] - data[state]
    profit <- profit + reward

    # Update Q-table
    Q[state, action] <- Q[state, action] +
      alpha *
        (reward + gamma * max(Q[new_state, ]) - Q[state, action])

    state <- new_state
  }

  return(profit)
}

# Train the Q-learning agent
profit <- Q_learning_agent(data, n_episodes = 1000)
print(paste("Total profit after training:", profit))