# library(dplyr, TTR, quantmod, xts)
source("util_functions.R")

csv_folder <- 'Binance Data/'
files_list <- list.files(csv_folder, full.names = FALSE)
symbols <- sub(".csv", "", files_list)

results_df <- data.frame(Symbol  = character(),
                         Premium = numeric())

for (symbol in symbols){
  csv_path <- paste0(csv_folder, symbol, '.csv')
  OHLCV <- read.csv(csv_path,
                    colClasses = c("index" = "POSIXct"))
  
  # Initialize variables
  params <- list(
    cash = 10000,
    entry_price = 0,
    position = "none",  # Position can be "long", "short", or "none"
    take_profit_pc = 0.05,  # 5% take profit
    stop_loss_pc = 0.05,  # 5% stop loss
    profit = 0,
    commission = 0.002,  # 0.2% commission
    order_size = 0.999
  )
  
  # Calculate SMAs
  OHLCV$sma14 <- SMA(Cl(OHLCV), n = 14)
  OHLCV$sma200 <- SMA(Cl(OHLCV), n = 200)
  OHLCV$Signal <- get_signal(OHLCV)
  OHLCV <- backtesting(OHLCV, params)
  Premium <- evaluate_strategy(OHLCV)
  
  new_row <- data.frame(Symbol=symbol,
                        Premium=Premium)
  results_df <- rbind(results_df, new_row)
}