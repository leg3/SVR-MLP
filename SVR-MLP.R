# MLP MODEL
# Diagnostic ratio - UMCSI:VIX
#
# This script evaluates a grid of simple feed-forward neural nets (MLP) using
# leakage-safe preprocessing and NN-style metrics (MSE/RMSE/MAE) on validation
# and test splits for horizons h = 1 and h = 3.
#
# There are TWO nested loops:
#   (1) lag-window loop : iterate over each lag_window in lag_grid
#   (2) horizon loop    : iterate over each forecast horizon h in horizons
#
# Leakage safety rule (preprocessing):
#   Standardization parameters (mu, sd) are fit on TRAIN only, then applied to
#   val/test. No information from val/test is used to scale train.
#
# Interpretation note:
#   Models are trained/evaluated on SCALED log(SVR), but we also compute
#   "raw-scale" error metrics by inverse-transforming predictions back to the
#   original log(SVR) scale (y_raw, y_hat_raw).

# Libraries
library(pipewelder)
library(tidyverse)
library(lubridate)
library(keras3)

# Set seed
set.seed(599)

# Retreive data from FRED
volatility_series <- get_fred("VIXCLS", "1990-01-02", "2025-12-31")
sentiment_series <- get_fred("UMCSENT", "1990-01-01", "2025-12-31")

# Monthly mean of VIX (convert daily VIX to monthly average)
mean_volatility_series <- volatility_series %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month) %>%
  summarize(mean_value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  rename(date = month)

# Log transform the series
log_sentiment_series <- sentiment_series %>%
  mutate(log_value_sen = log(value))

# Log transform the monthly-mean volatility series
log_mean_volatility_series <- mean_volatility_series %>%
  mutate(log_value_mnvol = log(mean_value))

# Join and compute transformed ratio
log_diagnostic_ratio_series <- log_sentiment_series %>%
  inner_join(log_mean_volatility_series, by = "date") %>%
  select(-value, -mean_value) %>%
  mutate(log_ratio_raw = (log_value_sen - log_value_mnvol))

# Partitioning (monthly obs)
n_test <- 84   # ~7 years

# Create modeling dataframe (monthly, ordered, no missing y)
df_all <- log_diagnostic_ratio_series %>%
  select(date, y = log_ratio_raw) %>%
  arrange(date) %>%
  filter(!is.na(y))

# Total number of observations
n <- nrow(df_all)

# Sanity check: need enough observations to have train + test
stopifnot(n_test < n)

# Compute global index for the start of the test block
i_test_start <- n - n_test + 1

# Subset df_all into training and test sets
train_df <- df_all[1:(i_test_start - 1), ]
test_df  <- df_all[i_test_start:n, ]

# Fit scaler parameters on TRAIN only (leakage-safe)
scaler_mu <- mean(train_df$y, na.rm = TRUE)
scaler_sd <- sd(train_df$y, na.rm = TRUE)

# Sanity check: scaler must be finite and sd must be > 0
stopifnot(is.finite(scaler_mu), is.finite(scaler_sd), scaler_sd > 0)

# Scale each split using TRAIN mu/sd
train_y_scaled <- (train_df$y - scaler_mu) / scaler_sd
test_y_scaled  <- (test_df$y  - scaler_mu) / scaler_sd

# Overwrite y for downstream modeling (all modeling uses scaled y)
train_df$y <- train_y_scaled
test_df$y  <- test_y_scaled

# make_supervised()
# Given a univariate series, build:
#   X : matrix of lagged windows (rows = forecast origins; cols = lagged values)
#   y : vector of targets at the specified forecast horizon
#
# Window ordering: oldest -> newest within the lag window
make_supervised <- function(series_values,
                            lag_window,
                            forecast_horizon) {
  n_observations <- length(series_values)

  # Forecast origin indices (t) that allow a full lag window and a future target
  forecast_origins <- lag_window:(n_observations - forecast_horizon)

  # Construct 2D design matrix
  X_mat <- t(sapply(forecast_origins, function(t_index) {
    series_values[(t_index - lag_window + 1):t_index]
  }))

  # Construct target vector at t + h
  y_vec <- sapply(forecast_origins, function(t_index) {
    series_values[t_index + forecast_horizon]
  })

  list(X = as.matrix(X_mat), y = as.numeric(y_vec))
}

# make_pred_tbl()
# Run model predictions for a supervised split and align predictions to dates.
# Optionally inverse-transform y and y_hat back to the original log(SVR) scale.
make_pred_tbl <- function(model,
                          sup,
                          y_vec,
                          split_dates,
                          lag_window,
                          h,
                          which_split = c("val", "test"),
                          scaler_mu = NULL,
                          scaler_sd = NULL) {
  which_split <- match.arg(which_split)

  # Predict and compute residuals (on the scaled modeling scale)
  y_hat <- as.numeric(model %>% predict(sup$X))
  y_vec <- as.numeric(y_vec)
  resid <- y_vec - y_hat

  # Align prediction targets to the correct monthly dates
  n_split <- length(split_dates)
  target_idx <- (lag_window:(n_split - h)) + h
  dates <- split_dates[target_idx]

  out <- tibble(
    date = dates,
    y = y_vec,
    y_hat = y_hat,
    resid = resid,
    lag_window = lag_window,
    horizon = h,
    split = which_split
  )

  # If scaler parameters are supplied, invert y and y_hat back to log(SVR)
  if (!is.null(scaler_mu) && !is.null(scaler_sd)) {
    out <- out %>%
      mutate(
        y_raw     = y * scaler_sd + scaler_mu,
        y_hat_raw = y_hat * scaler_sd + scaler_mu,
        resid_raw = y_raw - y_hat_raw
      )
  }

  return(out)
}
