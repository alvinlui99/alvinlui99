get_signal <- function(OHLCV) {
  signal <- rep(0, nrow(OHLCV))  # Initialize the signal vector with zeros
  
  # Check the conditions for the signal
  buy_signal <- 
    lag(OHLCV$sma14) < lag(OHLCV$sma200) &
    OHLCV$sma14 > OHLCV$sma200
  sell_signal <-
    lag(OHLCV$sma14) > lag(OHLCV$sma200) &
    OHLCV$sma14 < OHLCV$sma200
  
  signal[buy_signal] <- 1  # Set buy signal where conditions are met
  signal[sell_signal] <- -1  # Set sell signal where conditions are met
  
  return(signal)
}

close_position <- function(OHLCV, i, price){
  # i is the index (row) position of OHLCV
  
  # ignore cost when close position
  transfer <- OHLCV$Quantity_BOP[i] * price
  OHLCV$Quantity_EOP[i] <- 0
  OHLCV$Cash[i] <- OHLCV$Cash[i] + transfer
  
  return(OHLCV)
}

open_position <- function(OHLCV, i = integer(), price,
    tp = numeric(), sl = numeric(), size = numeric(1),
    commission = numeric(0)){
  # allow fractional qty
  budget <- OHLCV$Cash[i] * size
  qty <- budget / (price * (1 + commission))
  
  OHLCV$Cash[i] <- OHLCV$Cash[i] - budget
  OHLCV$Quantity_EOP[i] <- OHLCV$Quantity_EOP[i] + qty
  OHLCV$Take_Profit[i] <- tp
  OHLCV$Stop_Loss[i] <- sl
}

check_take_profit <- function(OHLCV, i, status){
  if (is.na(status$tp)) {
    return (FALSE)
  } else if (status$qty > 0 & OHLCV$High[i] > status$tp) {
    return (TRUE)
  } else if (status$qty < 0 & OHLCV$Low[i] < status$tp) {
    return (TRUE)
  }
  return (FALSE)
}

check_stop_loss <- function(OHLCV, i, status){
  if (is.na(status$sl)) {
    return (FALSE)
  } else if (status$qty > 0 & OHLCV$Low[i] < status$sl) {
    return (TRUE)
  } else if (status$qty < 0 & OHLCV$High[i] > status$sl) {
    return (TRUE)
  }
  return (FALSE)
}

execute_take_profit <- function(OHLCV, i, status){
  OHLCV$Cash[i:nrow(OHLCV)] <- OHLCV$Cash[i] + status$qty * status$tp
  status$qty <- 0
  status$tp <- status$sl <- NA
  OHLCV$qty[i:nrow(OHLCV)] <- rep(status$qty, nrow(OHLCV) - i + 1)
  
  return(list(OHLCV = OHLCV, status = status))
}

execute_stop_loss <- function(OHLCV, i, status){
  OHLCV$Cash[i:nrow(OHLCV)] <- OHLCV$Cash[i] + status$qty * status$sl
  status$qty <- 0
  status$tp <- status$sl <- NA
  OHLCV$qty[i:nrow(OHLCV)] <- rep(status$qty, nrow(OHLCV) - i + 1)
  
  return(list(OHLCV = OHLCV, status = status))
}

backtesting <- function(OHLCV, params){
  # Initialize columns with zeros----
  OHLCV$Cash <- rep(params$cash, nrow(OHLCV))
  
  OHLCV$qty <- 
    OHLCV$AV <- 
    rep(0, nrow(OHLCV))
  
  status <- list(
    active_order = "none",
    qty = 0,
    tp = NA,
    sl = NA
  )
  
  # Loop through the OHLCV data----
  for (i in 1:nrow(OHLCV)) {  # Start from the 201st row to ensure enough data for SMAs
    # First check if hit take profit or stop loss
    # Then check if there is active order
    # Then check signal to create order
    
    
    ## Take Profit or Stop Loss----
    # Check if hit take profit or stop loss
    
    if (check_take_profit(OHLCV, i, status)){
      temp <- execute_take_profit(OHLCV, i, status)
      OHLCV <- temp$OHLCV
      status <- temp$status
    }
    
    if (check_stop_loss(OHLCV, i, status)){
      temp <- execute_stop_loss(OHLCV, i, status)
      OHLCV <- temp$OHLCV
      status <- temp$status
    }
    
    # Check if opposite signal
    
    # Long position
    if (status$qty > 0){
      if(OHLCV$Signal[i] == -1){  # Short signal
        # Close position with Close price
        OHLCV$Cash[i:nrow(OHLCV)] <- OHLCV$Cash[i] + status$qty * Cl(OHLCV)[i]  
        status$qty <- 0
        status$tp <- status$sl <- NA
        OHLCV$qty[i:nrow(OHLCV)] <- 0
      }
    }
    
    # Short position
    if (status$qty < 0){  
      if(OHLCV$Signal[i] == 1){  # Short signal
        # Close position with Close price
        OHLCV$Cash[i:nrow(OHLCV)] <- OHLCV$Cash[i] + status$qty * Cl(OHLCV)[i]  
        status$qty <- 0
        status$tp <- status$sl <- NA
        OHLCV$qty[i:nrow(OHLCV)] <- 0
      }
    }
    
    ## Execute order----
    
    # Execute Long Market Order
    if (status$active_order == "Long Market Order"){
      entry_price <- OHLCV$Open[i]
      status$tp <- entry_price * (1 + params$take_profit_pc)
      status$sl <- entry_price * (1 - params$stop_loss_pc)
      
      budget <- OHLCV$Cash[i] * params$order_size
      status$qty <- budget / (entry_price * (1 + params$commission))
      
      cash <- OHLCV$Cash[i] - budget
      OHLCV$Cash[i:nrow(OHLCV)] <- cash
      OHLCV$qty[i:nrow(OHLCV)] <- status$qty
      
      status$active_order <- "none"
    }
    
    # Execute Short Market Order
    if (status$active_order == "Short Market Order"){
      entry_price <- OHLCV$Open[i]
      status$tp <- entry_price * (1 - params$take_profit_pc)
      status$sl <- entry_price * (1 + params$stop_loss_pc)
      
      budget <- OHLCV$Cash[i] * params$order_size
      status$qty <- - budget / (entry_price * (1 + params$commission))
      
      cash <- OHLCV$Cash[i] + budget
      OHLCV$Cash[i:nrow(OHLCV)] <- cash
      OHLCV$qty[i:nrow(OHLCV)] <- status$qty
      
      status$active_order <- "none"
    }
    
    
    ## Create order----
    
    # Create Long Market Order
    if (OHLCV$Signal[i] == 1 & status$qty == 0) {
      status$active_order <- "Long Market Order"
    }
    
    # Create Short Market Order
    if (OHLCV$Signal[i] == -1 & status$qty == 0) {
      status$active_order <- "Short Market Order"
    }
    
  }
  
  # Calculate Account Value----
  
  OHLCV$AV <- OHLCV$Cash + OHLCV$qty * Cl(OHLCV)
  
  return (OHLCV)
}

evaluate_strategy <- function(OHLCV){
  total_days <- as.numeric(OHLCV$index[nrow(OHLCV)] - OHLCV$index[1],
                           units = "days")
  Return_BnH <- (OHLCV$Close[nrow(OHLCV)] / OHLCV$Close[1]) ^
    (365 / total_days)
  Return_Strategy <- (OHLCV$AV[nrow(OHLCV)] / OHLCV$AV[1]) ^
    (365 / total_days)
  Premium <- Return_Strategy - Return_BnH
  
  return(Premium)
}

feature_generation <- function(OHLCV){
  # MACD----
  MACD_df <- as.data.frame(MACD(Cl(OHLCV)))
  names(MACD_df) <- paste0("ind_MACD_", names(MACD_df))
  OHLCV <- cbind(OHLCV, MACD_df)
  
  # RSI----
  OHLCV$ind_RSI <- RSI(Cl(OHLCV))
  
  # Moving Average----
  for (window in seq(20, 200, by=20)){
    sma <- SMA(Cl(OHLCV), window)
    OHLCV[[paste0("ind_SMA", window)]] <- sma / Cl(OHLCV)
    
    ema <- EMA(Cl(OHLCV), window)
    OHLCV[[paste0("ind_EMA", window)]] <- ema / Cl(OHLCV)
  }
  
  # Bollinger Bands----
  BBands_df <- as.data.frame(BBands(OHLCV[,c("High","Low","Close")]))
  names(BBands_df) <- paste0("ind_BBands_", names(BBands_df))
  OHLCV <- cbind(OHLCV, BBands_df)
  BBands_col_names_relative <- c("ind_BBands_dn", "ind_BBands_mavg", "ind_BBands_up")
  OHLCV[, c(BBands_col_names_relative, "Close")] <- OHLCV %>%
    select(all_of(c(BBands_col_names_relative, "Close"))) %>%
    mutate(across(all_of(BBands_col_names_relative), ~./Close))
  
  # Welles Wilder’s Directional Movement Index----
  ADX_df <- as.data.frame(ADX(OHLCV[,c("High","Low","Close")]))
  names(ADX_df) <- paste0("ind_ADX_", names(ADX_df))
  OHLCV <- cbind(OHLCV, ADX_df)
  
  # Aroon----
  aroon_df <- as.data.frame(aroon(OHLCV[,c("High","Low")]))
  names(aroon_df) <- paste0("ind_aroon_", names(aroon_df))
  OHLCV <- cbind(OHLCV, aroon_df)
  
  # Commodity Channel Index----
  CCI_df <- as.data.frame(CCI(OHLCV[,c("High","Low","Close")]))
  names(CCI_df) <- paste0("ind_CCI_", names(CCI_df))
  OHLCV <- cbind(OHLCV, CCI_df)
  
  
  # Volume----
  OHLCV$ind_Volume <- OHLCV$Volume
  
  # Return----
  OHLCV$ind_return <- c(NA, OHLCV$Close[-1] / OHLCV$Close[-nrow(OHLCV)])
  
  return(OHLCV)
}



