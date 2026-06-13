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