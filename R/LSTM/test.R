# Function to create 3D sequence for LSTM
create_lstm_sequence <- function(df = data.frame,
                                 timestep = integer(0)) {
  n <- nrow(df)
  p <- ncol(df)
  sequence <- array(0, dim = c(n - timestep + 1, timestep, p))

  for (i in seq_len(n - timestep + 1)) {
    for (j in seq_len(p)) {
      sequence[i, , j] <- df[i:(i + timestep - 1), j]
    }
  }

  return(sequence)
}