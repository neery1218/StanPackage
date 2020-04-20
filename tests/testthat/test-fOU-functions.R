require(rstan)
require(mvtnorm)
require(rethinking)
require(matrixcalc)
require(numDeriv)
library(tidyverse)


test_that("fou_mu drift term ",{

  H <- 0.9
  y_n <- 3
  gamma <- 0.9
  mu <- 1

  expect_equal(fou_mu(y_n, list(gamma=gamma, mu=mu, H=H)), -1.8)
})

test_that("fou_sigma Constant value test ", {

  H <- 0.9
  gamma <- 0.9
  mu <- 1

  expect_equal(fou_sigma(theta=list(gamma=gamma, mu=mu, H=H)),1)

})


test_that("fou_sim & csde_sim Simulate data ",{

  delta_t <- 1
  N <- 50
  H <- 0.9
  X0 <- 3
  gamma <- 0.9
  mu <- 1
  K <- 3

  data <- fOU_sim(N, list(gamma=gamma, mu=mu, H=H), X0, delta_t)
  expect_equal(length(data$Xt), N+1)
})

test_that("fou_logdens csde_logdens Log Density", {

  # Generating data
  N <- 10
  delta_t <- 1
  theta <- list(mu=0, gamma=1, H=0.5)
  sigma <-1

  dG <- rep(1, N)
  Xt <- rep(NA, N + 1)

  Xt[1] = 0
  for (i in 1:N) {
    Xt[i + 1] = Xt[i] + fou_mu(Xt[i], theta) * delta_t + dG[i]
  }

  #calculating log likelihood
  dX <- diff(Xt)
  mu <- fou_mu(Xt[1:N], theta)
  dG <- (dX - mu * delta_t) / sigma
  NTz <- SuperGauss::NormalToeplitz$new(N)
  ld <- NTz$logdens(dG, acf = fou_gamma(theta=theta,1, N))
  ld <- ld - N * log(sigma)

  #adding priors
  ld <- ld + dunif(theta$H, 0, 1, log = TRUE) + dunif(theta$gamma, 0, 2, log = TRUE) + dnorm(theta$mu, 0, 10, log = TRUE)


  #comparing against function
  ld_fou <- fou_logdens(Xt, delta_t, list(gamma=theta$gamma, mu=theta$mu, H=theta$H))
  expect_equal(ld, ld_fou)

})

# Prediction tests
test_that("getDgs", {
  # Simulate Xt, then verify the dGs returned by get_dGs is identical.
  N <- 10
  delta_t <- 1
  theta <- list(mu=0, gamma=1, H=0.5)

  dG <- rep(1, N)
  Xt <- rep(NA, N + 1)

  Xt[1] = 0
  for (i in 1:N) {
    Xt[i + 1] = Xt[i] + fou_mu(Xt[i], theta) * delta_t + dG[i]
  }

  fOU_data = list(X0=0, delta_t=delta_t, Xt=Xt, theta=theta)

  dGs_actual <- testproject1::get_dGs(fOU_data, theta)
  expect_equal(dG, dGs_actual)
})

# commented out because the test actually fits a stan model
# test_that("fou_predict Predicting and Plotting",{
#
#   delta_t <- 1
#   N <- 10
#   H <- 0.9
#   X0 <- 3
#   gamma <- 0.9
#   mu <- 1
#   K <- 3
#
#   data <- fOU_sim(N, list(gamma=gamma, mu=mu, H=H), X0, delta_t)
#   fit <- fit_fOU_process(data,1)
#   predicts <- fOU_predict(fit, X0, delta_t, 3, 3)
#   plot_prediction_interval(data, fit, delta_t, 4, 3)
# })

