library(TTR)
library(dplyr)
library(caret)
library(quantmod)
library(keras)
library(tensorflow)
library(ggplot2)


get_signal <- function(ohlcv_data) {
  signal <- rep(0, nrow(ohlcv_data))  # Initialize the signal vector with zeros
  # Check the conditions for the signal
  buy_signal <-
    lag(ohlcv_data$sma14) < lag(ohlcv_data$sma200) &
    ohlcv_data$sma14 > ohlcv_data$sma200
  sell_signal <-
    lag(ohlcv_data$sma14) > lag(ohlcv_data$sma200) &
    ohlcv_data$sma14 < ohlcv_data$sma200
  signal[buy_signal] <- 1  # Set buy signal where conditions are met
  signal[sell_signal] <- -1  # Set sell signal where conditions are met
  return(signal)
}

close_position <- function(ohlcv_data, i, status, price) {
  # i is the index (row) position of ohlcv_data
  # ignore cost when close position
  ohlcv_data$Cash[i:nrow(ohlcv_data)] <-
    ohlcv_data$Cash[i] + status$qty * Cl(ohlcv_data)[i]
  status$qty <- 0
  status$tp <- status$sl <- NA
  ohlcv_data$qty[i:nrow(ohlcv_data)] <- 0

  return(list(ohlcv_data = ohlcv_data, status = status))
}

execute_long_market_order <- function(ohlcv_data,
                                      i = integer(),
                                      status,
                                      params) {
  entry_price <- ohlcv_data$Open[i]
  status$tp <- entry_price * (1 - params$take_profit_pc)
  status$sl <- entry_price * (1 + params$stop_loss_pc)

  # Allow fraction
  budget <- ohlcv_data$Cash[i] * params$order_size
  status$qty <- - budget / (entry_price * (1 + params$commission))

  cash <- ohlcv_data$Cash[i] + budget
  ohlcv_data$Cash[i:nrow(ohlcv_data)] <- cash
  ohlcv_data$qty[i:nrow(ohlcv_data)] <- status$qty

  status$active_order <- "none"

  return(list(ohlcv_data = ohlcv_data, status = status))
}

execute_short_market_order <- function(ohlcv_data,
                                       i = integer(),
                                       status,
                                       params) {
  entry_price <- ohlcv_data$Open[i]
  status$tp <- entry_price * (1 - params$take_profit_pc)
  status$sl <- entry_price * (1 + params$stop_loss_pc)

  budget <- ohlcv_data$Cash[i] * params$order_size
  status$qty <- - budget / (entry_price * (1 + params$commission))

  cash <- ohlcv_data$Cash[i] + budget
  ohlcv_data$Cash[i:nrow(ohlcv_data)] <- cash
  ohlcv_data$qty[i:nrow(ohlcv_data)] <- status$qty

  status$active_order <- "none"

  return(list(ohlcv_data = ohlcv_data, status = status))
}

check_take_profit <- function(ohlcv_data, i, status) {
  if (is.na(status$tp)) {
    return(FALSE)
  } else if (status$qty > 0 && ohlcv_data$High[i] > status$tp) {
    return(TRUE)
  } else if (status$qty < 0 && ohlcv_data$Low[i] < status$tp) {
    return(TRUE)
  }
  return(FALSE)
}

check_stop_loss <- function(ohlcv_data, i, status) {
  if (is.na(status$sl)) {
    return(FALSE)
  } else if (status$qty > 0 && ohlcv_data$Low[i] < status$sl) {
    return(TRUE)
  } else if (status$qty < 0 && ohlcv_data$High[i] > status$sl) {
    return(TRUE)
  }
  return(FALSE)
}

execute_take_profit <- function(ohlcv_data, i, status) {
  ohlcv_data$Cash[i:nrow(ohlcv_data)] <-
    ohlcv_data$Cash[i] + status$qty * status$tp
  status$qty <- 0
  status$tp <- status$sl <- NA
  ohlcv_data$qty[i:nrow(ohlcv_data)] <-
    rep(status$qty, nrow(ohlcv_data) - i + 1)

  return(list(ohlcv_data = ohlcv_data, status = status))
}

execute_stop_loss <- function(ohlcv_data, i, status) {
  ohlcv_data$Cash[i:nrow(ohlcv_data)] <-
    ohlcv_data$Cash[i] + status$qty * status$sl
  status$qty <- 0
  status$tp <- status$sl <- NA
  ohlcv_data$qty[i:nrow(ohlcv_data)] <-
    rep(status$qty, nrow(ohlcv_data) - i + 1)

  return(list(ohlcv_data = ohlcv_data, status = status))
}

backtesting <- function(ohlcv_data, params) {
  # Initialize columns with zeros----
  ohlcv_data$Cash <- rep(params$cash, nrow(ohlcv_data))

  ohlcv_data$qty <-
    ohlcv_data$AV <-
    rep(0, nrow(ohlcv_data))

  status <- list(
    active_order = "none",
    qty = 0,
    tp = NA,
    sl = NA
  )

  bt_info <- list(ohlcv_data = ohlcv_data,
                  status     = status)

  # Loop through the ohlcv_data data----
  for (i in seq_len(nrow(bt_info$ohlcv_data))) {
    # First check if hit take profit or stop loss
    # Then check if there is active order
    # Then check signal to create order

    ## Take Profit or Stop Loss----
    # Check if hit take profit or stop loss

    if (check_take_profit(bt_info$ohlcv_data, i, bt_info$status)) {
      bt_info <- execute_take_profit(bt_info$ohlcv_data, i, bt_info$status)
    }

    if (check_stop_loss(bt_info$ohlcv_data, i, bt_info$status)) {
      bt_info <- execute_stop_loss(bt_info$ohlcv_data, i, bt_info$status)
    }

    # Check if opposite signal

    # Long position
    if (bt_info$status$qty > 0) {
      if (bt_info$ohlcv_data$Signal[i] == -1) {  # Short signal
        # Close position with Close price
        bt_info <- close_position(bt_info$ohlcv_data, i,
                                  bt_info$status, Cl(bt_info$ohlcv_data)[i])
      }
    }

    # Short position
    if (bt_info$status$qty < 0) {
      if (bt_info$ohlcv_data$Signal[i] == 1) {  # Short signal
        # Close position with Close price
        bt_info <- close_position(bt_info$ohlcv_data, i,
                                  bt_info$status, Cl(bt_info$ohlcv_data)[i])
      }
    }

    ## Execute order----

    # Execute Long Market Order
    if (bt_info$status$active_order == "Long Market Order") {
      bt_info <- execute_long_market_order(bt_info$ohlcv_data, i,
                                           bt_info$status, params)
    }

    # Execute Short Market Order
    if (bt_info$status$active_order == "Short Market Order") {
      bt_info <- execute_short_market_order(bt_info$ohlcv_data, i,
                                            bt_info$status, params)
    }


    ## Create order----
    if (bt_info$ohlcv_data$Signal[i] == 1 && bt_info$status$qty == 0) {
      bt_info$status$active_order <- "Long Market Order"
    } else if (bt_info$ohlcv_data$Signal[i] == -1 && bt_info$status$qty == 0) {
      bt_info$status$active_order <- "Short Market Order"
    }
  }

  # Vectorized Operations
  # Calculate Account Value----

  bt_info$ohlcv_data$AV <- bt_info$ohlcv_data$Cash +
    bt_info$ohlcv_data$qty * Cl(bt_info$ohlcv_data)

  return(bt_info$ohlcv_data)
}

evaluate_strategy <- function(ohlcv_data) {
  total_days <-
    as.numeric(ohlcv_data$index[nrow(ohlcv_data)] - ohlcv_data$index[1],
               units = "days")
  return_buy_and_hold <-
    (ohlcv_data$Close[nrow(ohlcv_data)] / ohlcv_data$Close[1]) ^
    (365 / total_days)
  return_strategy <- (ohlcv_data$AV[nrow(ohlcv_data)] / ohlcv_data$AV[1]) ^
    (365 / total_days)
  premium <- return_strategy - return_buy_and_hold

  return(premium)
}

feature_generation <- function(ohlcv_data) {
  # MACD----
  macd_df <- as.data.frame(MACD(Cl(ohlcv_data)))
  names(macd_df) <- paste0("ind_MACD_", names(macd_df))
  ohlcv_data <- cbind(ohlcv_data, macd_df)

  # RSI----
  ohlcv_data$ind_RSI <- RSI(Cl(ohlcv_data))

  # Moving Average----
  for (window in seq(20, 200, by = 20)) {
    sma <- SMA(Cl(ohlcv_data), window)
    ohlcv_data[[paste0("ind_SMA", window)]] <- sma / Cl(ohlcv_data)

    ema <- EMA(Cl(ohlcv_data), window)
    ohlcv_data[[paste0("ind_EMA", window)]] <- ema / Cl(ohlcv_data)
  }

  # Bollinger Bands----
  bbands_df <- as.data.frame(BBands(ohlcv_data[, c("High", "Low", "Close")]))
  names(bbands_df) <- paste0("ind_BBands_", names(bbands_df))
  ohlcv_data <- cbind(ohlcv_data, bbands_df)
  bbands_col_names_relative <-
    c("ind_BBands_dn", "ind_BBands_mavg", "ind_BBands_up")
  ohlcv_data[, c(bbands_col_names_relative, "Close")] <- ohlcv_data %>%
    select(all_of(c(bbands_col_names_relative, "Close"))) %>%
    mutate(across(all_of(bbands_col_names_relative), ~. / Close))

  # Welles Wilder’s Directional Movement Index----
  adx_df <- as.data.frame(ADX(ohlcv_data[, c("High", "Low", "Close")]))
  names(adx_df) <- paste0("ind_ADX_", names(adx_df))
  ohlcv_data <- cbind(ohlcv_data, adx_df)

  # Aroon----
  aroon_df <- as.data.frame(aroon(ohlcv_data[, c("High", "Low")]))
  names(aroon_df) <- paste0("ind_aroon_", names(aroon_df))
  ohlcv_data <- cbind(ohlcv_data, aroon_df)

  # Commodity Channel Index----
  cci_df <- as.data.frame(CCI(ohlcv_data[, c("High", "Low", "Close")]))
  names(cci_df) <- paste0("ind_CCI_", names(cci_df))
  ohlcv_data <- cbind(ohlcv_data, cci_df)

  # Volume----
  ohlcv_data$ind_Volume <- ohlcv_data$Volume

  # Return----
  ohlcv_data$ind_return <-
    c(NA, ohlcv_data$Close[-1] / ohlcv_data$Close[-nrow(ohlcv_data)])

  return(ohlcv_data)
}

lstm_oneoff_get_signal <- function(ohlcv_data,
                                   buy_threshold = numeric(0),
                                   sell_threshold = numeric(0)) {
  # Initialize the signal vector with zeros
  signal <- rep(0, NROW(ohlcv_data))

  model_input <- get_input_for_model(data = ohlcv_data,
                                     train_split = 0.8)

  model <- get_model(model_input)

  # Generate predictions
  predictions <- predict(model, model_input$x_test)

  # Check the conditions for the signal
  buy_signal <-
    predictions > buy_threshold
  sell_signal <-
    predictions < sell_threshold
  signal[buy_signal] <- 1  # Set buy signal where conditions are met
  signal[sell_signal] <- -1  # Set sell signal where conditions are met
  return(signal)
}

get_input_for_model <- function(data,
                                train_split = numeric(0)) {
  # 1. split train and test
  # 2. standardise
  # 3. pca
  # 4. reshape
  data_ind <- data %>%
    feature_generation() %>%
    na.omit() %>%
    select(starts_with("ind_"))

  data_ind$target <- c(data_ind$ind_return[-1], NA)
  data_ind <- na.omit(data_ind)

  # X

  ## 1. split train and test
  train_index <- createDataPartition(data_ind$target,
                                     p = train_split,
                                     list = FALSE)
  x_train <- data_ind[train_index, !names(data_ind) %in% "target"]
  x_test  <- data_ind[-train_index, !names(data_ind) %in% "target"]

  ## 2. standardise
  means <- colMeans(x_train)
  sds   <- apply(x_train, 2, sd)

  x_train <- as.data.frame(scale(x_train, center = means, scale = sds))
  x_test  <- as.data.frame(scale(x_test, center = means, scale = sds))

  ## 3. pca
  pca <- prcomp(x_train, scale = FALSE)

  x_train <- predict(pca, x_train)[, 1:8]
  x_test  <- predict(pca, x_test)[, 1:8]

  ## 4. reshape
  x_train <- array(x_train,
                   dim = c(dim(x_train)[1], 1, dim(x_train)[2]))

  x_test  <- array(x_test,
                   dim = c(dim(x_test)[1], 1, dim(x_test)[2]))

  # y

  ## 1. split train and test
  y_train <- data_ind[train_index, "target"]
  y_test  <- data_ind[-train_index, "target"]

  ## 2. reshape
  y_train <- array(y_train,
                   dim = c(length(y_train), 1))

  y_test  <- array(y_test,
                   dim = c(length(y_test), 1))

  return(list(x_train = x_train,
              y_train = y_train,
              x_test  = x_test,
              y_test  = y_test,
              pca     = pca))
}

get_model <- function(data,
                      val_split = numeric(0)) {
  model <- keras_model_sequential()

  model %>%
    layer_lstm(units = 50, input_shape = c(1, 8)) %>%
    layer_dense(units = 1)

  # Compile the model
  model %>% compile(
    loss = "mean_squared_error",
    optimizer = optimizer_adam()
  )

  # Train the LSTM model (Assuming you have target data model_input_y)
  model %>% fit(
    data$x_train, data$y_train,
    validation_split = val_split,
    epochs = 100,
    batch_size = 32
  )

  return(model)
}

backtesting_bu <- function(ohlcv_data, params) {
  # Initialize columns with zeros----
  ohlcv_data$Cash <- rep(params$cash, nrow(ohlcv_data))

  ohlcv_data$qty <-
    ohlcv_data$AV <-
    rep(0, nrow(ohlcv_data))

  status <- list(
    active_order = "none",
    qty = 0,
    tp = NA,
    sl = NA
  )

  # Loop through the ohlcv_data data----
  for (i in seq_len(nrow(ohlcv_data))) {
    # First check if hit take profit or stop loss
    # Then check if there is active order
    # Then check signal to create order

    ## Take Profit or Stop Loss----
    # Check if hit take profit or stop loss

    if (check_take_profit(ohlcv_data, i, status)) {
      temp <- execute_take_profit(ohlcv_data, i, status)
      ohlcv_data <- temp$ohlcv_data
      status <- temp$status
    }

    if (check_stop_loss(ohlcv_data, i, status)) {
      temp <- execute_stop_loss(ohlcv_data, i, status)
      ohlcv_data <- temp$ohlcv_data
      status <- temp$status
    }

    # Check if opposite signal

    # Long position
    if (status$qty > 0) {
      if (ohlcv_data$Signal[i] == -1) {  # Short signal
        # Close position with Close price
        temp <- close_position(ohlcv_data, i, status, Cl(ohlcv_data)[i])
        ohlcv_data <- temp$ohlcv_data
        status <- temp$status
      }
    }

    # Short position
    if (status$qty < 0) {
      if (ohlcv_data$Signal[i] == 1) {  # Short signal
        # Close position with Close price
        temp <- close_position(ohlcv_data, i, status, Cl(ohlcv_data)[i])
        ohlcv_data <- temp$ohlcv_data
        status <- temp$status
      }
    }

    ## Execute order----

    # Execute Long Market Order
    if (status$active_order == "Long Market Order") {
      temp <- execute_long_market_order(ohlcv_data, i, status, params)
      ohlcv_data <- temp$ohlcv_data
      status <- temp$status
    }

    # Execute Short Market Order
    if (status$active_order == "Short Market Order") {
      temp <- execute_short_market_order(ohlcv_data, i, status, params)
      ohlcv_data <- temp$ohlcv_data
      status <- temp$status
    }


    ## Create order----
    if (ohlcv_data$Signal[i] == 1 && status$qty == 0) {
      status$active_order <- "Long Market Order"
    } else if (ohlcv_data$Signal[i] == -1 && status$qty == 0) {
      status$active_order <- "Short Market Order"
    }
  }

  # Vectorized Operations
  # Calculate Account Value----

  ohlcv_data$AV <- ohlcv_data$Cash + ohlcv_data$qty * Cl(ohlcv_data)

  return(ohlcv_data)
}