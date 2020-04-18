require(rstan) # observe startup messages
require(mvtnorm)
library(matrixcalc)
library(tidyverse)
require(ggplot2)
require(gridExtra)

fou_mu <- function(y_n, theta) {
  -1 * theta$gamma * (y_n - theta$mu)
}

fou_gamma <- function(theta, dt, n){
  gamma <- rep(0, n)
  H <- theta$H
  for (i in 1:n) {
    gamma[i] <- (dt^(2*H)/2) * (abs(i)^(2*H) + abs(i - 2)^(2*H) - 2*abs(i - 1)^(2*H))
  }
  gamma
}

fou_sigma <- function(theta, dt, N) {
  1
}

# thanks to mlysy for writing this function!
#' Generate cSDE observations.
#'
#' @param X0 Initial cSDE value at time `t = 0`.
#' @param fft Whether to use fast (but sometimes less stable) FFT simulation method.  See [SuperGauss::rnormtz()].
#' @return A vector of `N+1` cSDE observations recorded at intervals of `dt` starting from `X0`.
csde_sim <- function(N, dt, X0, theta,
                     mu_fun, sigma_fun, gamma_fun, fft = TRUE) {
  # dXt = X_{t-1} + mu_fun(X_{t-1}, theta) * dt + sigma * dG

  sig <- sigma_fun(theta) # diffusion
  gam <- gamma_fun(theta, dt, N) # autocorrelation
  dG <- SuperGauss::rnormtz(acf = gam, fft = fft) # noise increments
  Xt <- rep(NA, N+1) # cSDE time series

  Xt[1] <- X0 # initialize
  for(ii in 1:N) {
    # recursion
    Xt[ii + 1] <- Xt[ii] + mu_fun(Xt[ii], theta) * dt + sig * dG[ii]
  }
  Xt
}

# given parameters, simulates data from the fOU(theta).
fOU_sim <- function(N, theta, X0, delta_t) {
  Xt <- csde_sim(N, delta_t, X0, theta, fou_mu, fou_sigma, fou_gamma)
  list(
    Xt = Xt,
    theta = theta,
    X0 = X0,
    delta_t = delta_t
  )
}

# returns a stanfit object that's fitted to c(X0, Xt)
fit_fOU_process <- function(Xt, K, iter = 2e3, control = list(adapt_delta = 0.95)) {
  N <- length(Xt) - 1  # remove X0

  # when we're using k-level approximations, the stan model
  # requires us to specify which timesteps have data, and which timesteps don't have data.
  ii_obs <- seq(K, N * K, K)
  ii_mis <- setdiff(seq(1, N * K), ii_obs)

  fit <- rstan::sampling(
    stanmodels$test_fou_process,
    data = list(
      Xt = tail(Xt, N),
      N = N,
      K = K,
      delta_t = delta_t,
      X0 = X0,
      ii_obs = ii_obs,
      ii_mis = ii_mis
    ),
    control = control,
    iter = iter,
    init = function() { list(H=fOU_data$theta$H, gamma=fOU_data$theta$gamma, mu=fOU_data$theta$mu)} # added nice initial values to make things converge faster.
  )
  fit
}

fit_fOU_multiple_K <- function(Xt, Ks) {
  all_samples <- data.frame()

  fits <- sapply(Ks, function(K) {
    fit_fOU_process(Xt, K)
  })

  for (i in 1:length(Ks)) {
    post_samples <- rstan::extract(fits[[i]])

    param_samples <- as.data.frame(post_samples)[c('H', 'gamma', 'mu')]
    param_samples$K <- Ks[i]
    all_samples <- rbind(all_samples, param_samples)
  }
  all_samples <- as_tibble(all_samples)
  list(post_samples=all_samples, fits=fits)
}

# thanks to mlysy for writing this function!
csde_logdens <- function(Xt, dt, theta,
                         mu_fun, sigma_fun, gamma_fun) {
  dX <- diff(Xt)
  N <- length(dX)
  mu <- mu_fun(Xt[1:N], theta) # drift
  sig <- sigma_fun(theta) # diffusion
  gam <- gamma_fun(theta, dt, N) # autocorrelation
  dG <- (dX - mu * dt) / sig # noise increments
  NTz <- SuperGauss::NormalToeplitz$new(N) # instantiate NTz distribution
  ld <- NTz$logdens(dG, acf = gam)
  ld - N * log(sig) # jacobian for change-of-variables dX <-> dG
}

# wrapper around csde_logdens for the fOU process
fou_logdens <- function(Xt, delta_t, theta) {
  csde_logdens(Xt, delta_t, theta, fou_mu, fou_sigma, fou_gamma) +
    dunif(theta$H, 0, 1, log = TRUE) + # H ~ uniform(1)
    dunif(theta$gamma, 0, 2, log = TRUE) +
    dnorm(theta$mu, 0, 10, log = TRUE)
}

# check that theta_hat is a global AND local maximum.
# This is done by essentially grid searching one parameter while holding everything else constant.
# This is expensive, but feasible since our parameter constraints are fairly tight.
plot_likelihoods <- function(fOU_data, fit, K, thresholds=list(mu=0.1, gamma=0.1, H = 0.05)) {
  post_samples <- rstan::extract(fit, pars=c("gamma", "mu", "H"))
  Xtk_samples <- rstan::extract(fit)$Xt_k
  Xtk_hat <- apply(Xtk_samples, 2, mean)

  mu_hat <- mean(post_samples$mu)
  H_hat <- mean(post_samples$H)
  gamma_hat <- mean(post_samples$gamma)
  theta_hat <- list(gamma = gamma_hat, mu = mu_hat, H = H_hat)
  delta_t <- fOU_data$delta_t / K

  print(theta_hat)
  print(delta_t)

  par(mfrow=c(2,3))

  # global checks
  range_H <- seq(0.0, 1, length.out = 1000)
  loglikelihoods <- sapply(range_H, function(H) { fou_logdens(Xtk_hat, delta_t, theta = list(mu=theta_hat$mu, gamma=theta_hat$gamma, H = H) )})
  plot(range_H, loglikelihoods)
  abline(v = theta_hat$H)


  range_gamma <- seq(0, 2, length.out = 1000)
  loglikelihoods <- sapply(range_gamma, function(gamma) { fou_logdens(Xtk_hat, delta_t, theta = list(mu=theta_hat$mu, gamma=gamma, H = theta_hat$H) )})
  plot(range_gamma, loglikelihoods)
  abline(v = theta_hat$gamma)

  range_mu <- seq(-5, 5, length.out = 1000)
  loglikelihoods <- sapply(range_mu, function(mu) { fou_logdens(fOU_data$Xt, delta_t, theta = list(mu=mu, gamma=theta_hat$gamma, H = theta_hat$H) )})
  plot(range_mu, loglikelihoods)
  abline(v = theta_hat$mu)

  # local checks
  range_H <- seq(theta_hat$H - thresholds$H, theta_hat$H + thresholds$H, length.out = 1000)
  loglikelihoods <- sapply(range_H, function(H) { fou_logdens(Xtk_hat, delta_t, theta = list(mu=theta_hat$mu, gamma=theta_hat$gamma, H = H) )})
  plot(range_H, loglikelihoods)
  abline(v = theta_hat$H)

  range_gamma <- seq(theta_hat$gamma - thresholds$gamma, theta_hat$gamma + thresholds$gamma, length.out = 1000)
  loglikelihoods <- sapply(range_gamma, function(gamma) { fou_logdens(Xtk_hat, delta_t, theta = list(mu=theta_hat$mu, gamma=gamma, H = theta_hat$H) )})
  plot(range_gamma, loglikelihoods)
  abline(v = theta_hat$gamma)

  range_mu <- seq(theta_hat$mu - thresholds$mu, theta_hat$mu + thresholds$mu, length.out = 1000)
  loglikelihoods <- sapply(range_mu, function(mu) { fou_logdens(fOU_data$Xt, delta_t, theta = list(mu=mu, gamma=theta_hat$gamma, H = theta_hat$H) )})
  plot(range_mu, loglikelihoods)
  abline(v = theta_hat$mu)
}

# given a stanfit object, predicts the next n data points using the parameters' posterior distributions
# given a stanfit object, draws parameters from the posterior distributions and simulates fOU data.
# the new fOU data is then aggregated to produce prediction confidence intervals.
# this is an expensive computation.
fOU_predict <- function(fit, X0, delta_t, n_points, n_samples) {

  # each row contains a sample path taken by the fitted fOU process
  pred_matrix <- matrix(nrow = n_samples, ncol = n_points + 1)
  posterior_samples <- rstan::extract(fit, pars=c("mu", "gamma", "H"))

  for (i in 1:n_samples) {
    # draw parameters from posterior distribution
    mu <- sample(posterior_samples$mu, 1)
    gamma <- sample(posterior_samples$gamma, 1)
    H <- sample(posterior_samples$H, 1)

    # sample path using parameters
    # FIXME: This is incorrect. the dG's calculated in fOU_sim below aren't correlated with the dG's from before.
    # To do that, we can't use fOU_sim.
    # This will work for H = 0.5, and we probably can't tell the different for H in [0.3, 0.7]
    # use condMVN here with the original dGs.
    fOU_data_pred <- fOU_sim(n_points, theta = list(mu=mu, gamma=gamma, H=H), X0, delta_t)
    pred_matrix[i, ] <- fOU_data_pred$Xt
  }

  pred_matrix
}


# given fOU_data, uses fOU_predict to generate confidence intervals, then plots n sample paths from the _true_ fOU parameters
plot_prediction_interval <- function(fOU_data, fit, delta_t, n_points, n_samples) {
  Xt <- tail(fOU_data$Xt, 20)
  N <- length(Xt)

  par(mfrow=c(1,1))
  plot(NULL, ylim=c(-20,20), xlim = c(1, N + n_points), xlab = "time (t)", ylab = "Xt", main = "Predictions with sample paths")
  points(1:N, Xt)
  cols = c("grey20", "grey50", "grey90")

  pred_matrix <- fOU_predict(fit, tail(fOU_data$Xt, 1), delta_t, n_points - 1, n_samples)
  CI <- apply(pred_matrix, 2, function(ts) { quantile(ts, probs = c(0.25, 0.75)) })

  # 90% confidence interval
  polygon(c(N:(N + n_points - 1), rev(N:(N + n_points - 1))), c(CI[1,], rev(CI[2,])), col = "grey90", border = NA)

  for (i in 1:10) {
    Xt <- fOU_sim(n_points - 1, fOU_data$theta, tail(fOU_data$Xt,1), fOU_data$delta_t)$Xt
    lines(N:(N + n_points - 1), Xt, col = "black")
  }
}

# given data and a stanfit object, plots Xt along with the computed missing data confidence intervals
plot_CI <- function(fOU_data, fit, K, N, title = "Xt and 99% CIs of interpolated points") {
  t_k <- (1 : (K * N)) / K

  CI_lower <- apply(as.matrix(rstan::extract(fit)$Xt_k), 2, function(ts) { quantile(ts, probs = c(0.01)) })
  CI_upper <- apply(as.matrix(rstan::extract(fit)$Xt_k), 2, function(ts) { quantile(ts, probs = c(0.99)) })

  plot(NULL, xlim=c(0, N), ylim=c(min(fOU_data$Xt[1:N]), max(fOU_data$Xt[1:N])), xlab="time (t)", ylab="X(t)", main = title)

  polygon(c(t_k, rev(t_k)), c(CI_upper[1:(K * N)], rev(CI_lower[1:(K * N)])), col = "grey90", border = NA)
  points(1:N, fOU_data$Xt[2:(N + 1)], col="black")
}

# given a tibble of posterior samples (from fit_fOU_process), calculate the
plot_param_posterior_distributions <- function(post_tb) {
  p1 <- ggplot(data=post_tb) +
    geom_density(mapping=aes(x=H, group=K_factor, color = K_factor))

  p2 <- ggplot(data=post_tb) +
    geom_density(mapping=aes(x=gamma, group=K_factor, color = K_factor))

  p3 <- ggplot(data=post_tb) +
    geom_density(mapping=aes(x=mu, group=K_factor, color = K_factor))
  grid.arrange(p1, p2, p3, nrow=3)
}

