require(rstan) # observe startup messages
require(mvtnorm)
require(matrixcalc)
require(tidyverse)
require(ggplot2)
require(gridExtra)


#' Drift term for the fOU process.
#'
#' @param y_n is a real number.
#' @param theta is a list with the parameters of the fOU process (where sigma is permanently 1):
#' \describe{
#'   \item{H}{Hurst index which is between 0 and 1.}
#'   \item{mu}{Mean of the fOU process which is greater than 0.}
#'   \item{gamma}{Mean reverting speed of the process which is greater than 0.}
#' }
#' @return A real number.
#' @export
#'
fou_mu <- function(y_n, theta) {
  -1 * theta$gamma * (y_n - theta$mu)
}

#' Autocorrelation function for the fOU process.
#'
#' @param theta is a list with the parameters of the fOU process (where sigma is permanently 1):
#' \describe{
#'   \item{H}{Hurst index which is between 0 and 1.}
#'   \item{mu}{Mean of the fOU process which is positive.}
#'   \item{gamma}{Mean reverting speed of the process which is positive.}
#'   }
#' @param dt is the interobservation time between each datapoint of the fOU process.
#' @param n Number of observations of the fOU process.
#' @return An `n` dimensional vector of covariances.
#' @details The toeplitz matrix of the autocorrelation function of the fOU process is defined by the following vector of covariances:
#' ```
#' V[i] = (abs(i)^(2H) + abs(i - 2)^(2H) - 2 x abs(i - 1)^(2H)
#'
#' ```
#' @export
fou_gamma <- function(theta, dt, n){
  gamma <- rep(0, n)
  H <- theta$H
  for (i in 1:n) {
    gamma[i] <- (dt^(2*H)/2) * (abs(i)^(2*H) + abs(i - 2)^(2*H) - 2*abs(i - 1)^(2*H))
  }
  gamma
}

#' Stochastic deviation for the fOU process.
#'
#' @param theta is a list with the parameters of the fOU process (where sigma is permanently 1):
#' \describe{
#'   \item{H}{Hurst index which is between 0 and 1.}
#'   \item{mu`}{Mean of the fOU process which is positive.}
#'   \item{gamma}{Mean reverting speed of the process which is positive.}
#'   }
#' @param dt is the interobservation time between each datapoint of the fOU process.
#' @param N Number of observations of the fOU process.
#' @return A real number.
#' @details Maintained constant at 1 to simplify calculations.
#' @export
fou_sigma <- function(theta, dt, N) {
  1
}

#' Generate cSDE observations.
#'
#' @param N Number of observations.
#' @param X0 Initial cSDE value at time `t = 0`.
#' @param dt The interobservation time between each observation of the cSDE.
#' @param theta List of parameters of the cSDE.
#' @param mu_fun Drift term of the cSDE which is a function of Xt and theta.
#' @param sigma_fun Stochastic deviationof the cSDE which is a function of theta.
#' @param gamma_fun Autocorrelation of the cSDE which is a function of theta, dt and N.
#' @param fft Whether to use fast (but sometimes less stable) FFT simulation method.  See [SuperGauss::rnormtz()].
#' @return A vector of `N+1` cSDE observations recorded at intervals of `dt` starting from `X0`.
#' @details Function provided by Prof. Lysy that simulates data generation for any coloured-noise stochastic differential equation.Tthanks to mlysy for writing this function!
#' @export
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

#' Generate fOU observations.
#'
#' @param N Number of observations to generate.
#' @param theta is a list with the parameters of the fOU process fOU process (where sigma is permanently 1):
#' \describe{
#'   \item{H}{Hurst index which is between 0 and 1.}
#'   \item{mu}{Mean of the fOU process which is positive.}
#'   \item{gamma}{Mean reverting speed of the process which is positive.}
#'   }
#' @param X0 Initial fOU value at time `t = 0`.
#' @param delta_t The interobservation time between each observation of the fOU process.
#' @return A vector of `N+1` fOU observations recorded at intervals of `dt` starting from `X0`.
#' @details Generates data for the fOU which is characterized by the following equation:
#' \deqn{
#' dX_t = \gamma(X_t - \mu)*\Delta t + \sigma B^{H}_t
#' }
#' @export
fOU_sim <- function(N, theta, X0, delta_t) {
  Xt <- csde_sim(N, delta_t, X0, theta, fou_mu, fou_sigma, fou_gamma)
  list(
    Xt = Xt,
    theta = theta,
    X0 = X0,
    delta_t = delta_t
  )
}

#' Fit an fOU process.
#'
#' Fits an fOU process to the given data using a pre-compiled stan mode.
#'
#' @param fOU_data is a list of
#' \describe{
#'   \item{Xt}{Vector of observations at level k.}
#'   \item{delta_t}{The interobservation time between each observation of the fOU process.}
#'   \item{X0}{Initial fOU value at time `t = 0`.}
#'   }
#' @param K The level of euler-approximation which is a positive number less than N.
#' @param iter Maximum number of total iterations.
#' @param control List of control parameter for rstan::sampling. See [rstan::sampling()] for more details.
#' @return A stanfit object that is fitted to `c(X0,Xt)`.
#' @export
fit_fOU_process <- function(fOU_data, K, iter = 2e3, control = list(adapt_delta = 0.95)) {
  Xt <- fOU_data$Xt
  delta_t <- fOU_data$delta_t
  X0 <- fOU_data$X0

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

#' Fit multiple levels of an fOU process.
#'
#' Fits an fOU process to the given data across multiple k-level approximations using a pre-compiled stan mode.
#'
#' @param fOU_data fOU_data object returned by fOU_sim.
#' @param Ks is vector of levels for the k-level approximations to be fitted.
#' @return A list of `Ks` stanfit objects that is fitted to `c(X0,Xt)` for each level k.
#' @export
fit_fOU_multiple_K <- function(fOU_data, Ks) {
  all_samples <- data.frame()

  fits <- sapply(Ks, function(K) {
    fit_fOU_process(fOU_data, K)
  })

  for (i in 1:length(Ks)) {
    post_samples <- rstan::extract(fits[[i]])

    param_samples <- as.data.frame(post_samples)[c('H', 'gamma', 'mu')]
    param_samples$K <- Ks[i]
    all_samples <- rbind(all_samples, param_samples)
  }
  all_samples <- dplyr::as_tibble(all_samples)
  list(post_samples=all_samples, fits=fits)
}


#' Log-density of cSDE observations.
#'
#' Calculates the log-density of `p(Xt | theta)`, where `Xt` are observations of a cSDE recorded at interobservation time `dt`.
#' Calculate log density of cSDE observations.
#'
#' @param Xt A vector of cSDE observations.
#' @param dt The interobservation time between each observation of the cSDE.
#' @param theta List of parameters of the cSDE
#' @param mu_fun Drift term of the cSDE which is a function of Xt and theta.
#' @param sigma_fun Stochastic deviationof the cSDE which is a function of theta.
#' @param gamma_fun Autocorrelation of the cSDE which is a function of theta, dt and N.
#' @details Calculates the log-density of `p(Xt | theta)`, where `Xt` are observations of a cSDE recorded at interobservation time `dt`. Thanks to mlysy for writing this function!
#' @return A scalar containing the log-density of the cSDE evaluated at its arguments.
#' @export
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


#' Log-density of fOU observations.
#'
#' @param Xt A vector of fOU observations.
#' @param delta_t is the interobservation time between each observation of the fOU process.
#' @param theta is a list with the parameters of the fOU process fOU process (where sigma is permanently 1):
#' \describe{
#'   \item{H}{Hurst index which is between 0 and 1.}
#'   \item{mu}{Mean of the fOU process which is positive.}
#'   \item{gamma}{Mean reverting speed of the process which is positive.}
#'   }
#' @return A scalar containing the log-density of the fOU evaluated at its arguments.
#' @export
fou_logdens <- function(Xt, delta_t, theta) {
  csde_logdens(Xt, delta_t, theta, fou_mu, fou_sigma, fou_gamma) +
    dunif(theta$H, 0, 1, log = TRUE) + # H ~ uniform(1)
    dunif(theta$gamma, 0, 2, log = TRUE) +
    dnorm(theta$mu, 0, 10, log = TRUE)
}

#' Plot likelihoods of fOU parameters.
#'
#' Plots log likelihoods for each of the fOU parameters - H, gamma and mu.
#'
#' @param fOU_data is a list of
#' \describe{
#'   \item{Xt}{Vector of observations at level k.}
#'   \item{delta_t}{The interobservation time between each observation of the fOU process.}
#'   \item{X0}{Initial fOU value at time `t = 0`.}
#'   }
#' @param fit Stan fit object fitted on the `fOU_data`.
#' @param K The level of euler-approximation which is a positive number less than N.
#' @param thresholds List of thresholds for mu, gamma and H to perturb each term to grid search over.
#' @return Plot of the likelihoods of each parameter.
#' @details Check that theta_hat is a global AND local maximum.This is done by essentially grid searching one parameter while holding everything else constant.This is expensive, but feasible since our parameter constraints are fairly tight.
#' @export
plot_likelihoods <- function(fOU_data, fit, K, thresholds=list(mu=0.1, gamma=0.1, H = 0.05)) {
  post_samples <- rstan::extract(fit, pars=c("gamma", "mu", "H"))
  Xtk_samples <- rstan::extract(fit)$Xt_k
  Xtk_hat <- apply(Xtk_samples, 2, mean)

  mu_hat <- mean(post_samples$mu)
  H_hat <- mean(post_samples$H)
  gamma_hat <- mean(post_samples$gamma)
  theta_hat <- list(gamma = gamma_hat, mu = mu_hat, H = H_hat)
  delta_t <- fOU_data$delta_t / K

  # global checks
  range_H <- seq(0.0, 1, length.out = 1000)
  loglikelihoods <- sapply(range_H, function(H) { fou_logdens(Xtk_hat, delta_t, theta = list(mu=theta_hat$mu, gamma=theta_hat$gamma, H = H) )})
  plot(range_H, loglikelihoods, xlab = "H", main = "Global L(H|Xt)")
  abline(v = theta_hat$H, col = "red")

  range_gamma <- seq(0, 2, length.out = 1000)
  loglikelihoods <- sapply(range_gamma, function(gamma) { fou_logdens(Xtk_hat, delta_t, theta = list(mu=theta_hat$mu, gamma=gamma, H = theta_hat$H) )})
  plot(range_gamma, loglikelihoods, xlab = "gamma", main = "Global L(gamma|Xt)")
  abline(v = theta_hat$gamma, col = "red")

  range_mu <- seq(-5, 5, length.out = 1000)
  loglikelihoods <- sapply(range_mu, function(mu) { fou_logdens(fOU_data$Xt, delta_t, theta = list(mu=mu, gamma=theta_hat$gamma, H = theta_hat$H) )})
  plot(range_mu, loglikelihoods, xlab = "mu", main = "Global L(mu|Xt)")
  abline(v = theta_hat$mu, col = "red")

  # local checks
  range_H <- seq(theta_hat$H - thresholds$H, theta_hat$H + thresholds$H, length.out = 1000)
  loglikelihoods <- sapply(range_H, function(H) { fou_logdens(Xtk_hat, delta_t, theta = list(mu=theta_hat$mu, gamma=theta_hat$gamma, H = H) )})
  plot(range_H, loglikelihoods, xlab = "H", main = "Local L(H|Xt)")
  abline(v = theta_hat$H, col = "red")

  range_gamma <- seq(theta_hat$gamma - thresholds$gamma, theta_hat$gamma + thresholds$gamma, length.out = 1000)
  loglikelihoods <- sapply(range_gamma, function(gamma) { fou_logdens(Xtk_hat, delta_t, theta = list(mu=theta_hat$mu, gamma=gamma, H = theta_hat$H) )})
  plot(range_gamma, loglikelihoods, xlab= "gamma", main = "Local L(gamma|Xt)")
  abline(v = theta_hat$gamma, col = "red")

  range_mu <- seq(theta_hat$mu - thresholds$mu, theta_hat$mu + thresholds$mu, length.out = 1000)
  loglikelihoods <- sapply(range_mu, function(mu) { fou_logdens(fOU_data$Xt, delta_t, theta = list(mu=mu, gamma=theta_hat$gamma, H = theta_hat$H) )})
  plot(range_mu, loglikelihoods, xlab = "mu", main = "Local L(mu|Xt)")
  abline(v = theta_hat$mu, col = "red")
}

#' Given fOU_data, calculates the observed fBM increments wrt theta
#'
#' @param fOU_data fOU_data object returned from fOU_sim
#' @param theta parameters used to generate fOU_data
#' @return Vector of points corresponding to the fBM increments
#' @export
get_dGs <- function(fOU_data, theta) {
  Xt <- fOU_data$Xt
  dX <- diff(Xt)
  N <- length(dX)
  mu <- fou_mu(Xt[1:N], theta) # drift
  sig <- fou_sigma(theta) # diffusion
  gam <- fou_gamma(theta, fOU_data$delta_t, N) # autocorrelation
  dG <- (dX - mu * fOU_data$delta_t) / sig # noise increments
  dG
}

#' Draws a dG_future sample from the distribution dG_future | dG_obs
#' @param dG_obs observed dG values
#' @param n_points number of dG points to sample
#' @param theta sampled parameters from the posterior distribution
#' @param delta_t delta_t
#' @return sampled dG values at time [t, .... t + n_points]
#' @export
get_dGs_future <- function(dGs_obs, n_points, theta, delta_t) {
  N <- length(dGs_obs)

  big_toep <- toeplitz(fou_gamma(theta, delta_t, N + n_points))

  sigma_11 <- big_toep[1:n_points, 1:n_points]
  sigma_12 <- big_toep[1:n_points, (n_points+1):(N+n_points)]
  sigma_21 <- big_toep[(n_points+1):(N+n_points), 1:n_points]

  sigma_22 <- big_toep[(n_points+1):(N+n_points), (n_points+1):(N+n_points)]

  # dg_future | dg_obs= g ~ Normal(sigma_12 * inverse(sigma_22) * g, sigma_11 - sigma_12 * inverse(sigma_22) * sigma_21)
  # TODO: use solveV
  inv_sigma_22 <- solve(sigma_22)
  cond_sigma <- sigma_11 - sigma_12 %*% inv_sigma_22 %*% sigma_21
  cond_mu <- sigma_12 %*% solve(sigma_22) %*% dGs_obs

  dGs_future <- mvtnorm::rmvnorm(1, cond_mu, cond_sigma)
  dGs_future
}


#' Make predictions using an fOU fit.
#'
#' Given a Stan fit object, it predicts the next n data points using the parameters' posterior distributions.
#'
#' @param fOU_data fOU_data object returned from fOU_sim
#' @param fit Stan fit object of an fOU process.
#' @param X0 Initial fOU value at time `t = 0`.
#' @param delta_t Interobservation time between each observation of the fOU process.
#' @param n_points Number of points to predict.
#' @param n_samples Number of sample paths to simulate.
#' @return list object with the observed dG values (dG_obs | theta = theta_sample), future dG values (dG_future | dG_obs), and the predictions (Xt_future)
#' @details Draws parameters from the posterior distributions and simulates fOU data. The new fOU data is then aggregated to produce prediction confidence intervals. This is an expensive computation.
#' @export
fOU_predict <- function(fOU_data, fit, X0, delta_t, n_points, n_samples) {

  # each row contains a sample path taken by the fitted fOU process
  dGs_obs_matrix <- matrix(nrow = n_samples, ncol = length(fOU_data$Xt) - 1)
  dGs_future_matrix <- matrix(nrow = n_samples, ncol = n_points)
  pred_matrix <- matrix(nrow = n_samples, ncol = n_points + 1)
  posterior_samples <- rstan::extract(fit, pars=c("mu", "gamma", "H"))


  for (i in 1:n_samples) {
    # draw parameters from posterior distribution
    mu <- sample(posterior_samples$mu, 1)
    gamma <- sample(posterior_samples$gamma, 1)
    H <- sample(posterior_samples$H, 1)
    theta <- list(mu=mu, gamma=gamma, H=H)

    # get observed dGs
    dGs_obs <- get_dGs(fOU_data, theta)
    dGs_future <- get_dGs_future(dGs_obs, n_points, theta, fOU_data$delta_t)

    dGs_future_matrix[i,] <- dGs_future
    dGs_obs_matrix[i,] <- dGs_obs

    sig <- fou_sigma(theta, delta_t, n_points) # hardcoded to one

    # construct Xt_future
    Xt_future <- rep(0, n_points+1) # cSDE time series
    Xt_future[1] <- X0 # initialize

    for(ii in 1:n_points) {
      # recursion
      Xt_future[ii + 1] <- Xt_future[ii] + fou_mu(Xt_future[ii], theta) * delta_t + sig * dGs_future[ii]
    }

    pred_matrix[i, ] <- Xt_future
  }
  list(pred_matrix=pred_matrix, dGs_future_matrix=dGs_future_matrix, dGs_obs_matrix=dGs_obs_matrix)
}


#' Plot prediction intervals.
#'
#' Given fOU_data, uses fOU_predict to generate confidence intervals, then plots n sample paths from the `true`` fOU parameters.
#' @param fOU_data is a list of
#' \describe{
#'   \item{Xt}{Vector of observations at level k.}
#'   \item{delta_t}{The interobservation time between each observation of the fOU process.}
#'   \item{X0}{Initial fOU value at time `t = 0`.}
#'   }
#' @param fit Stan fit object of an fOU process.
#' @param delta_t Interobservation time between each observation of the fOU process.
#' @param n_points Number of points to predict.
#' @param n_samples Number of sample paths to simulate.
#' @return Plot of prediction intervals of each of the fOU parameters.
#' @export
plot_prediction_interval <- function(fOU_data, pred_matrix) {
  Xt <- tail(fOU_data$Xt, 20)
  N <- length(Xt)
  n_points <- ncol(pred_matrix)

  plot(NULL, ylim=c(-20,20), xlim = c(1, N + n_points), xlab = "time (t)", ylab = "Xt", main = "Predictions with sample paths")
  points(1:N, Xt)

  # 90% confidence interval
  CI <- apply(pred_matrix, 2, function(ts) { quantile(ts, probs = c(0.05, 0.95)) })
  polygon(c(N:(N + n_points - 1), rev(N:(N + n_points - 1))), c(CI[1,], rev(CI[2,])), col = "slateblue1", border = NA)

  # 50% confidence interval
  CI <- apply(pred_matrix, 2, function(ts) { quantile(ts, probs = c(0.25, 0.75)) })
  polygon(c(N:(N + n_points - 1), rev(N:(N + n_points - 1))), c(CI[1,], rev(CI[2,])), col = "slateblue4", border = NA)
}

#' Plot confindence intervals.
#'
#' Given fOU_data, uses fOU_predict to generate confidence intervals, then plots n sample paths from the `true`` fOU parameters.
#' @param fOU_data is a list of
#' \describe{
#'   \item{Xt}{Vector of observations at level k.}
#'   \item{delta_t}{The interobservation time between each observation of the fOU process.}
#'   \item{X0}{Initial fOU value at time `t = 0`.}
#'   }
#' @param fit Stan fit object of an fOU process.
#' @param K The level of euler-approximation which is a positive number less than N.
#' @param N Number of observations of the fOU process.
#' @param title Title of the plot.
#' @return Plot of confidence intervals of each of the fOU parameters.
#' @export
plot_CI <- function(fOU_data, fit, K, N, title = "Xt and 99% CIs of interpolated points") {
  t_k <- (1 : (K * N)) / K

  CI_lower <- apply(as.matrix(rstan::extract(fit)$Xt_k), 2, function(ts) { quantile(ts, probs = c(0.01)) })
  CI_upper <- apply(as.matrix(rstan::extract(fit)$Xt_k), 2, function(ts) { quantile(ts, probs = c(0.99)) })

  plot(NULL, xlim=c(0, N), ylim=c(min(fOU_data$Xt[1:N]), max(fOU_data$Xt[1:N])), xlab="time (t)", ylab="X(t)", main = title)

  polygon(c(t_k, rev(t_k)), c(CI_upper[1:(K * N)], rev(CI_lower[1:(K * N)])), col = "grey90", border = NA)
    points(1:N, fOU_data$Xt[2:(N + 1)], col="black")
}

#' Plot posterior distribution for parameters.
#'
#' Given a tibble of posterior samples (from fit_fOU_process), calculate the posterior distribution and plot them.
#'
#' @param post_tb Tibble of posterior samples.
#' @param fOU_data data object returned by fOU_sim
#' @return Plot of the posterior distribution of each parameter of the fOU process.
#' @export
plot_param_posterior_distributions <- function(post_tb, fOU_data) {
  p1 <- ggplot(data=post_tb) +
    geom_density(mapping=aes(x=post_tb$H, group=post_tb$K, color = post_tb$K)) +
    geom_vline(xintercept = fOU_data$theta$H)

  p2 <- ggplot(data=post_tb) +
    geom_density(mapping=aes(x=post_tb$gamma, group=post_tb$K, color = post_tb$K)) +
    geom_vline(xintercept = fOU_data$theta$gamma)

  p3 <- ggplot(data=post_tb) +
    geom_density(mapping=aes(x=post_tb$mu, group=post_tb$K, color = post_tb$K)) +
    geom_vline(xintercept = fOU_data$theta$mu)

  grid.arrange(p1, p2, p3, nrow=3)
}

