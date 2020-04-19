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

test_that("fit_fou Fitting Data and plot", {

  delta_t <- 1
  N <- 3
  H <- 0.9
  X0 <- 3
  gamma <- 0.9
  mu <- 1
  K <- 3

  data <- fOU_sim(N, list(gamma=gamma, mu=mu, H=H), X0, delta_t)
  tibbl <- fit_fOU_multiple_K(data, c(1,2))
  plot_likelihoods(data, tibbl$fits[[1]], K=1)
  plot_CI(data,tibbl$fits[[1]], 1, 3)

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

test_that("fou_predict Predicting and Plotting",{

  delta_t <- 1
  N <- 10
  H <- 0.9
  X0 <- 3
  gamma <- 0.9
  mu <- 1
  K <- 3

  data <- fOU_sim(N, list(gamma=gamma, mu=mu, H=H), X0, delta_t)
  fit <- fit_fOU_process(data,1)
  predicts <- fOU_predict(fit, X0, delta_t, 3, 3)
  plot_prediction_interval(data, fit, delta_t, 4, 3)
})

