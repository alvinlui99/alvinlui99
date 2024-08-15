library(TTR)
library(dplyr)

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

close_position <- function(ohlcv_data, i, price) {
  # i is the index (row) position of ohlcv_data
  # ignore cost when close position
  transfer <- ohlcv_data$Quantity_BOP[i] * price
  ohlcv_data$Quantity_EOP[i] <- 0
  ohlcv_data$Cash[i] <- ohlcv_data$Cash[i] + transfer
  return(ohlcv_data)
}

open_position <- function(ohlcv_data,
                          i = integer(),
                          price,
                          tp = numeric(),
                          sl = numeric(),
                          size = numeric(1),
                          commission = numeric(0)) {
  # allow fractional qty
  budget <- ohlcv_data$Cash[i] * size
  qty <- budget / (price * (1 + commission))
  ohlcv_data$Cash[i] <- ohlcv_data$Cash[i] - budget
  ohlcv_data$Quantity_EOP[i] <- ohlcv_data$Quantity_EOP[i] + qty
  ohlcv_data$Take_Profit[i] <- tp
  ohlcv_data$Stop_Loss[i] <- sl
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
        ohlcv_data$Cash[i:nrow(ohlcv_data)] <-
          ohlcv_data$Cash[i] + status$qty * Cl(ohlcv_data)[i]
        status$qty <- 0
        status$tp <- status$sl <- NA
        ohlcv_data$qty[i:nrow(ohlcv_data)] <- 0
      }
    }

    # Short position
    if (status$qty < 0) {
      if (ohlcv_data$Signal[i] == 1) {  # Short signal
        # Close position with Close price
        ohlcv_data$Cash[i:nrow(ohlcv_data)] <-
          ohlcv_data$Cash[i] + status$qty * Cl(ohlcv_data)[i]
        status$qty <- 0
        status$tp <- status$sl <- NA
        ohlcv_data$qty[i:nrow(ohlcv_data)] <- 0
      }
    }

    ## Execute order----

    # Execute Long Market Order
    if (status$active_order == "Long Market Order") {
      entry_price <- ohlcv_data$Open[i]
      status$tp <- entry_price * (1 + params$take_profit_pc)
      status$sl <- entry_price * (1 - params$stop_loss_pc)

      budget <- ohlcv_data$Cash[i] * params$order_size
      status$qty <- budget / (entry_price * (1 + params$commission))

      cash <- ohlcv_data$Cash[i] - budget
      ohlcv_data$Cash[i:nrow(ohlcv_data)] <- cash
      ohlcv_data$qty[i:nrow(ohlcv_data)] <- status$qty

      status$active_order <- "none"
    }

    # Execute Short Market Order
    if (status$active_order == "Short Market Order") {
      entry_price <- ohlcv_data$Open[i]
      status$tp <- entry_price * (1 - params$take_profit_pc)
      status$sl <- entry_price * (1 + params$stop_loss_pc)

      budget <- ohlcv_data$Cash[i] * params$order_size
      status$qty <- - budget / (entry_price * (1 + params$commission))

      cash <- ohlcv_data$Cash[i] + budget
      ohlcv_data$Cash[i:nrow(ohlcv_data)] <- cash
      ohlcv_data$qty[i:nrow(ohlcv_data)] <- status$qty

      status$active_order <- "none"
    }


    ## Create order----

    # Create Long Market Order
    if (ohlcv_data$Signal[i] == 1 && status$qty == 0) {
      status$active_order <- "Long Market Order"
    }

    # Create Short Market Order
    if (ohlcv_data$Signal[i] == -1 && status$qty == 0) {
      status$active_order <- "Short Market Order"
    }
  }

  # Calculate Account Value----

  ohlcv_data$AV <- ohlcv_data$Cash + ohlcv_data$qty * Cl(ohlcv_data)

  return(ohlcv_data)
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

