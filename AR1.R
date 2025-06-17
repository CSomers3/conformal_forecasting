# Load necessary libraries
library(forecast)
library(tsibble)
library(dplyr)

# Simulate AR(1) process
set.seed(123)
n <- 10000
phi <- 0.7
y <- arima.sim(model = list(ar = phi), n = n)

# Split into training (fit AR), calibration (for residuals), and test (for prediction)
n_train <- 7000
n_calib <- 2000
n_test <- n - n_train - n_calib

train_data <- y[1:n_train]
calib_data <- y[(n_train + 1):(n_train + n_calib)]
test_data <- y[(n_train + n_calib + 1):n]

# Fit AR model on training data
fit <- arima(train_data, order = c(1, 0, 0))  # AR(1)

# Get predictions for calibration window
calib_preds <- predict(fit, n.ahead = n_calib)$pred
calib_residuals <- abs(calib_data - calib_preds)

# Set desired coverage
alpha <- 0.1  # 90% prediction interval
quantile_e <- quantile(calib_residuals, probs = 1 - alpha)

# Predict on test set
test_preds <- predict(fit, n.ahead = n_test)$pred

# Construct conformal prediction intervals
lower_bound <- test_preds - quantile_e
upper_bound <- test_preds + quantile_e

# Combine into a data frame for inspection
results <- data.frame(
  Time = (n_train + n_calib + 1):n,
  True = test_data,
  Pred = test_preds,
  Lower = lower_bound,
  Upper = upper_bound
)

# Compute coverage
coverage <- mean(results$True >= results$Lower & results$True <= results$Upper)
print(paste("Empirical coverage:", round(coverage, 3)))

# Plot
library(ggplot2)
ggplot(results, aes(x = Time)) +
  geom_line(aes(y = True), color = 'black') +
  geom_line(aes(y = Pred), color = 'blue') +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.3) +
  ggtitle("Conformal Prediction Intervals for AR(1) Model")
