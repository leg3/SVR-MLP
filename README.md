# SVR-MLP

Multilayer perceptron forecasting workflow for the log-transformed Sentiment–Volatility Ratio.

## Overview

This repository preserves the multilayer perceptron model developed for the capstone project:

> **Predictive Models for the Diagnostic Ratio of Consumer Sentiment and Volatility**

The project constructs the Sentiment–Volatility Ratio, or SVR, from two economic indicators:

* University of Michigan Consumer Sentiment Index (`UMCSENT`)
* CBOE Volatility Index (`VIXCLS`)

The MLP uses lagged observations of the log-transformed SVR to generate direct forecasts at one-month and three-month horizons.

This repository currently represents the preserved research implementation. It is not yet a production package, automated forecasting service, or deployment-ready application.

## Research context

The SVR combines household sentiment with financial-market volatility:

[
SVR_t = \frac{UMCSI_t}{VIX_t}
]

The modeled series is the natural logarithm of the ratio:

[
\log(SVR_t) = \log(UMCSI_t) - \log(VIX_t)
]

Daily VIX observations are aggregated to monthly means before the ratio is calculated. UMCSI is already reported monthly.

The historical capstone dataset covered January 1990 through December 2025 and contained 432 monthly observations.

## Repository scope

This repository contains only the MLP forecasting workflow.

It does not currently include:

* AR or ARIMA models
* LSTM models
* structural-break analysis
* exploratory data analysis
* Shiny applications
* automated data refreshes
* model deployment
* scheduled retraining
* CI/CD configuration
* saved Keras model artifacts
* forecast visualizations

The absence of a plotting section is intentional. The MLP was evaluated as a candidate model but was not selected as the final overall model used for the capstone forecast figures.

## Data sources

The workflow retrieves both source series from the Federal Reserve Economic Database through `pipewelder::get_fred()`.

| Series    | Description                                     | Frequency used                    |
| --------- | ----------------------------------------------- | --------------------------------- |
| `UMCSENT` | University of Michigan Consumer Sentiment Index | Monthly                           |
| `VIXCLS`  | CBOE Volatility Index                           | Daily, aggregated to monthly mean |

Requested retrieval windows:

* VIX: January 2, 1990 through December 31, 2025
* UMCSI: January 1, 1990 through December 31, 2025

The two monthly series are joined by date using an inner join.

## Workflow

The preserved script performs the following stages.

### 1. Data acquisition

The UMCSI and VIX series are retrieved from FRED.

### 2. Monthly VIX aggregation

Daily VIX observations are grouped by calendar month and averaged:

```r
mean(value, na.rm = TRUE)
```

The logarithm is applied after monthly averaging. The workflow therefore calculates:

```text
log(monthly mean VIX)
```

rather than:

```text
monthly mean of daily log(VIX)
```

### 3. Log-SVR construction

UMCSI and monthly mean VIX are independently log-transformed and joined by month.

The modeled value is then constructed as:

```r
log_ratio_raw = log_value_sen - log_value_mnvol
```

### 4. Chronological partitioning

The final 84 monthly observations are reserved as the test partition.

The historical 432-observation dataset therefore produced:

| Partition | Observations |
| --------- | -----------: |
| Training  |          348 |
| Test      |           84 |
| Total     |          432 |

The script calculates the total observation count dynamically after joining the series and removing missing target values. It does not hard-code or assert that the current retrieval must contain exactly 432 observations.

### 5. Training-only scaling

Standardization parameters are estimated from the training partition only:

[
z_t = \frac{y_t-\mu_{train}}{\sigma_{train}}
]

The same training mean and standard deviation are then applied to the test partition.

This prevents test observations from influencing the scaling parameters used for model training.

### 6. Supervised lag-window construction

The `make_supervised()` helper converts the scaled univariate series into a conventional two-dimensional MLP feature matrix.

Each row contains a lagged sequence ordered from oldest to newest:

```text
[y(t-L+1), y(t-L+2), ..., y(t)]
```

The corresponding target is:

```text
y(t+h)
```

where:

* `L` is the lag-window length
* `h` is the forecast horizon

The MLP therefore receives a two-dimensional matrix:

```text
rows = forecast origins
columns = lagged monthly values
```

No LSTM-style sequence dimension is retained.

### 7. Direct forecasting

Both forecast horizons are implemented directly.

For horizon 1:

```text
lag window ending at t → target at t + 1
```

For horizon 3:

```text
lag window ending at t → target at t + 3
```

The three-month model does not recursively predict months `t + 1` and `t + 2`.

It also does not include observations from `t + 1` or `t + 2` in the feature window used to predict `t + 3`.

### 8. MLP training

A separate model is trained for every combination of lag window and forecast horizon.

The experiment grid contains:

```r
lag_grid <- c(24, 18, 15, 12, 9, 6, 3)
horizons <- c(1, 3)
```

This produces 14 model runs.

### 9. Test evaluation

After training, each model is evaluated against a supervised test matrix created from the test partition.

Predictions are aligned with their target dates and inverse-transformed from the standardized scale back to the log-SVR scale.

### 10. Result aggregation

Per-run metrics and prediction tables are combined after the experiment loop.

Configurations are ranked separately by horizon using test MAE on the restored log-SVR scale.

## MLP architecture

Each candidate uses the same feed-forward architecture:

```text
Input layer:  lag_window features
Hidden layer: 16 dense units, ReLU activation
Output layer: 1 dense unit, linear activation
```

The model predicts one scalar log-SVR value.

### Compilation

| Setting        | Value               |
| -------------- | ------------------- |
| Optimizer      | Adam                |
| Learning rate  | 0.001               |
| Loss           | Mean squared error  |
| Tracked metric | Mean absolute error |

No explicit dropout, weight decay, batch normalization, or other regularization is used.

The script does not explicitly override the default Keras weight initializer.

## Training configuration

| Setting                   |      Value |
| ------------------------- | ---------: |
| Maximum epochs            |        300 |
| Batch size                |         16 |
| Internal validation split |        20% |
| Early-stopping patience   |  20 epochs |
| Early-stopping monitor    | `val_loss` |
| Restore best weights      |        Yes |
| Shuffle observations      |         No |
| Console verbosity         |          2 |

The internal validation split is created by Keras from the supervised training data. No test observations are supplied to model fitting or early stopping.

The reported `best_epoch` is calculated as the position of the minimum validation loss in the recorded training history.

## Reproducibility

The preserved workflow applies several reproducibility controls.

### R seed

```r
set.seed(599)
```

### Initial Keras seed

```r
set_random_seed(599)
```

### Per-run seed

Before each lag-window and horizon combination, the Keras session is cleared and a deterministic run-specific seed is assigned:

```r
keras3::clear_session()
set_random_seed(599 + lag_window * 10 + h)
```

### Chronological fitting

Training uses:

```r
shuffle = FALSE
```

This preserves the existing row order during model fitting.

### Remaining sources of variation

Exact historical results are not guaranteed across environments. Differences can still arise from:

* R version
* Python version
* `keras3` version
* TensorFlow version
* Keras backend behavior
* CPU versus GPU execution
* hardware-specific numerical operations
* deterministic-operation settings
* package dependency changes
* revised FRED observations
* differences in the configured Python environment

The repository aims first to reproduce the historical workflow. Exact numerical reproduction may not be possible when the original software environment is unavailable.

## Metrics

Metrics are calculated on two scales.

### Scaled metrics

The following values come from Keras evaluation of the standardized target:

* `test_mse`
* `test_rmse`
* `test_mae`

### Restored log-SVR metrics

Predictions and actual values are inverse-transformed using the training mean and standard deviation before calculating:

* `test_mse_raw`
* `test_rmse_raw`
* `test_mae_raw`

In this script, the suffix `raw` means **unscaled log-SVR**.

It does not mean that values have been converted back to the original, non-logarithmic SVR ratio.

Model configurations are ranked using:

```text
test_mae_raw
```

## Historical baseline

The following values were reported in the capstone and are retained as a historical comparison baseline.

They should not be treated as values that the repository must be forced to reproduce.

| Horizon | Lag window | Best epoch |    Test RMSE |     Test MAE |
| ------: | ---------: | ---------: | -----------: | -----------: |
|       1 |          3 |        136 | 0.2233823997 | 0.1515181788 |
|       1 |          6 |         33 | 0.2336580691 | 0.1686646703 |
|       1 |          9 |         60 | 0.2313692362 | 0.1692234069 |
|       1 |         12 |         31 | 0.2644516359 | 0.1858166867 |
|       1 |         15 |         99 | 0.2001002695 | 0.1589566739 |
|       1 |         18 |        178 | 0.2189039267 | 0.1784414189 |
|       1 |         24 |        279 | 0.2405648674 | 0.1885487879 |
|       3 |          3 |         20 | 0.3849735919 | 0.2714175831 |
|       3 |          6 |         26 | 0.3503197395 | 0.2440399392 |
|       3 |          9 |        134 | 0.3718213900 | 0.2544132503 |
|       3 |         12 |        126 | 0.3924331664 | 0.2787531297 |
|       3 |         15 |         71 | 0.3037570240 | 0.2364661083 |
|       3 |         18 |         93 | 0.3310675463 | 0.2651792995 |
|       3 |         24 |        143 | 0.3335663637 | 0.2638680723 |

Historically, the lowest MLP test MAE values were:

|  Horizon | Selected lag window |     Test MAE |
| -------: | ------------------: | -----------: |
|  1 month |            3 months | 0.1515181788 |
| 3 months |           15 months | 0.2364661083 |

## Output objects

The script creates several important R objects.

### `results_mlp`

Combined metrics for every lag-window and horizon combination.

Columns include:

* `lag_window`
* `horizon`
* `best_epoch`
* `test_mse`
* `test_rmse`
* `test_mae`
* `test_mse_raw`
* `test_rmse_raw`
* `test_mae_raw`

### `preds_mlp`

Combined test prediction records across all experiments.

Prediction records include:

* forecast date
* scaled actual value
* scaled predicted value
* scaled residual
* lag window
* forecast horizon
* restored log-SVR actual value
* restored log-SVR prediction
* restored log-SVR residual

### `metrics_mlp_result`

The combined metrics table ranked by:

1. forecast horizon
2. restored log-SVR test MAE

## CSV export

The final ranked metrics table is written to:

```text
MLP Metrics Latest.csv
```

The file is created in the current R working directory.

The script does not currently export:

* prediction tables
* training histories
* fitted model files
* serialized weights
* plots

## Requirements

### R packages

The preserved workflow requires:

```r
library(pipewelder)
library(tidyverse)
library(lubridate)
library(keras3)
```

### External runtime

`keras3` requires a compatible Python and neural-network backend environment.

The historical script assumes that the Keras backend is already installed and configured. Exact original package and backend versions were not preserved in this repository.

### Network access

The script retrieves data from FRED at runtime and therefore requires internet access unless the acquisition step is later replaced with a preserved local dataset.

## Running the workflow

Run the primary R script from the repository root in an environment where all R and Keras dependencies are available.

The script executes sequentially:

1. retrieve the source data
2. construct monthly log-SVR
3. create the chronological split
4. fit the training scaler
5. construct supervised windows
6. train the 14 MLP configurations
7. evaluate test predictions
8. rank configurations
9. export the metrics CSV

Neural-network training output is printed to the console because:

```r
verbose = 2
```

## Important implementation behavior

### Rolling forecasts within the test block

The supervised test windows are constructed from `test_df$y` alone.

As a result:

* the first test prediction requires enough observations within the test block to fill the selected lag window;
* the test windows do not use the final training observations as initial lag context;
* the first evaluated forecast date changes with lag-window length and forecast horizon;
* configurations with different lag windows are evaluated over different numbers of target observations.

The models use earlier observed test values as lagged inputs for later test targets. This represents a rolling forecast evaluation with observed history rather than a single fixed-origin forecast across the entire test period.

### Test-based lag selection

Early stopping uses an internal validation split from the training data.

However, the final lag-window configuration within each horizon is selected by comparing test MAE across the candidate grid.

The test partition therefore serves both as:

* the out-of-sample evaluation period;
* the basis for selecting the preferred lag window.

This behavior is preserved from the research workflow and should be considered when interpreting model rankings.

### Dynamic source data

The requested dates are fixed, but FRED observations may be revised.

A future execution may therefore produce values that differ from the dataset used during the original capstone run.

### No fixed environment lock

The repository does not yet contain:

* `renv.lock`
* Python requirements
* TensorFlow version pinning
* a container image
* deterministic TensorFlow configuration
* hardware metadata

These may be considered later as separate reproducibility work.

## Validation priorities

Future validation work should examine:

* the exact joined monthly observation count
* missing or unmatched dates after the inner join
* FRED revision effects
* feature and target alignment for each horizon
* forecast-date alignment
* test sample counts by lag window
* validation-split behavior under the active Keras version
* reproducibility across repeated runs
* scaled versus restored metric consistency
* compatibility with a documented modern R, Python, Keras, and TensorFlow environment

Any modeling correction should be separated from formatting, documentation, or dependency changes.

## Development approach

Changes to this repository should remain narrow and reviewable.

The intended workflow is:

1. preserve the recovered research implementation;
2. validate each component as written;
3. document discrepancies between the paper and executed code;
4. make only minimal changes required for reproducibility;
5. separate cleanup from methodological corrections;
6. avoid combining unrelated changes in a single pull request.

The original implementation should not be overwritten or silently altered to reproduce the published metrics.

## Project status

The historical MLP workflow has been reconstructed incrementally through focused feature branches covering:

* FRED data acquisition and log-SVR construction
* chronological train-test partitioning
* training-only scaling
* supervised lag-window construction
* prediction-table and date alignment
* experiment-grid configuration
* MLP training and evaluation
* result aggregation and CSV export

The next stage is validation of the preserved workflow before substantive cleanup or modernization.
