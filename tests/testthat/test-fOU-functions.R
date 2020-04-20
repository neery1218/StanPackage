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

  delta_t <- 1
  N <- 10
  H <- 0.9
  X0 <- 3
  gamma <- 0.9
  mu <- 1
  K <- 3

  data <- fOU_sim(N, list(gamma=gamma, mu=mu, H=H), X0, delta_t)
  ld <- fou_logdens(data$Xt, delta_t, list(gamma=gamma, mu=mu, H=H))
  expect_equal(is.numeric(ld), TRUE)

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


test_that("future_dgs", {
  N <- 1
  dGs_obs <- c(0)
  theta <- list(mu=0, H=0.5)  # will produce an identity variance matrix.

  # given the parameters, we're sampling from N(0, 1)
  dGs_sample <- sapply(1:4e3, function(i) {get_dGs_future(dGs_obs, 1, theta, 1)})

  # likelihood of normal distribution
  obj_fun <- function(mu, sigma) {
    sum(-(mu - dGs_sample)^2 / sigma^2) - length(dGs_sample)/2 * log(sigma^2)
  }

  # make sure mu=0, sigma=1 is more likely than (mu=1, sigma=1), (mu=0, sigma=5)
  # this are very wide checks to make sure we don't get really unlucky.
  expect_gt(obj_fun(0,1), obj_fun(2,1))
  expect_gt(obj_fun(0,1), obj_fun(0,5))
})


# commented out because the test actually fits a stan model
# can't test fOU_predict without a stanfit object.
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

