rm(list=ls())

# Sample size
n = 250

# Number of Monte Carlo iterations
N = 10000

# level
alpha = 0.1

# Inclusion indicators and lengths
incl = rep(0,N)
len = rep(0,N)

# Monte Carlo Simulation Study

for(i in 1:N){
  # Seed for simulated data
  set.seed(i)
  
  # Covariates simulation
  x1 = rnorm(n,0,1)
  x2 = rnorm(n,0,1)
  x3 = rnorm(n,0,1)
  
  # errors simulation
  eps = rnorm(n,0,0.5)
  
  # outcomes simulation
  y = 1 + x1 + x2 + x3 + eps
  
  # Design matrix
  x <- cbind(x1,x2,x3)
  
  # Training and Calibration indexes
  ind_train <- sort(sample(1:n, n * 0.5, replace = FALSE))
  ind_calibration <- setdiff(1:n, ind_train)
  
  # Training and calibration sets
  y_train <- y[ind_train]
  X_train0 <- x[ind_train,]
  y_calibration <- y[ind_calibration]
  X_calibration0 <- x[ind_calibration,]
  
  # Index for non-conformity measure
  k <- ceiling((1-alpha)*(length(ind_calibration)+1))
  
  # Fitting training set
  linmod <- lm(y_train ~ X_train0, data = data.frame(y_train, X_train0))
  
  # Calculating non-conformity measures on the calibration set
  res_cal = abs(cbind(1,X_calibration0)%*%coef(linmod) - y_calibration)
  
  # Quantile for conformal prediction intervals
  q_lm = sort(res_cal)[k]

  # New covariate and outcome simulation
  set.seed(i+N)
  x_new <- rbind(rnorm(3))
  eps_new <- rnorm(1,0,0.5)
  colnames(x_new) <- colnames(X_train0)
  
  y_new = 1 + x_new[1] + x_new[2] + x_new[3] + eps_new
  
  # Predict the response
  y_pred <- coef(linmod)%*%c(1,x_new)
  
  
  # Conformal Prediction interval
  CPI_L <-  y_pred - q_lm
  
  CPI_U <- y_pred + q_lm
  
  # Checking inclusion and length of the CP interval
  incl[i] <-  ifelse( (CPI_L < y_new)&(y_new < CPI_U), 1, 0 )

  len[i] = CPI_U - CPI_L

  print(i)
  
}

# Empirical inclusion (marginal validity)
mean(incl)

# Length of CP intervals
plot(density(len), xlim = c(0,4))

summary(len)

