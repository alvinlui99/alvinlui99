source("util_functions.R")

interval <- "30m/"

csv_folder <- "Binance Data/"
files_list <- list.files(paste0(csv_folder, interval), full.names = FALSE)
symbols <- sub(".csv", "", files_list)

results_df <- data.frame(Symbol  = character(),
                         Premium = numeric())

for (symbol in symbols){
  csv_path <- paste0(csv_folder, interval, symbol, ".csv")
  ohlcv_data <- read.csv(csv_path,
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
    order_size = 0.999,
    buy_threshold  = 1.01,
    sell_threshold = 0.99
  )

  # Calculate
  ohlcv_data$Signal <- lstm_oneoff_get_signal(ohlcv_data,
                                              params$buy_threshold,
                                              params$sell_threshold)
  ohlcv_data <- backtesting(ohlcv_data, params)
  premium <- evaluate_strategy(ohlcv_data)

  new_row <- data.frame(Symbol = symbol,
                        Premium = premium)
  results_df <- rbind(results_df, new_row)
}