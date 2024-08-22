TradingEnvironment <- function(price_data, indicators) {
  # Class of trading environment
  # Contains market information
  # Allows interaction with the market

  env <- list(
    indicators = indicators,
    price_data = price_data,
    current_index = 1,
    position = 0,  # 0: no position, 1: holding stock
    initial_balance = 10000,
    balance = 10000,
    shares_held = 0
  )

  env$reset <- function() {
    # Reset everything back to time 0
    env$current_index <- 1
    env$balance <- env$initial_balance
    env$shares_held <- 0
    env$position <- 0
  }

  env$get_state <- function() {
    # It is used by the agent to make decision
    # Return the last (hist_length) indicators
    return(env$indicators[env$current_index, , , drop = FALSE])
  }

  env$step <- function(action) {
    current_price <- env$price_data[env$current_index]
    # Actions: 0 = Hold, 1 = Buy, 2 = Sell
    if (action == 1 && env$position == 0) {  # Buy
      env$shares_held <- env$balance / current_price
      env$balance <- 0
      env$position <- 1
    } else if (action == 2 && env$position == 1) {  # Sell
      env$balance <- env$shares_held * current_price
      env$shares_held <- 0
      env$position <- 0
    }

    # Move to the next time step
    current_index <- env$current_index
    env$current_index <- current_index + 1
    done <- env$current_index >= length(env$price_data)

    # Reward is the current balance minus initial balance
    reward <-
      env$balance +
      env$shares_held * current_price -
      env$initial_balance

    return(list(reward = reward, done = done))
  }

  return(env)
}