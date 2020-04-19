# Model
# likelihood: y ~ normal(test_fun(mu), 1)
# prior: mu ~ unif(0, 100)
options(mc.cores = parallel::detectCores()) # TODO: maybe reset after end of test?
library(mvtnorm)

#' A test function implemented to test custom gradient functionality.
#' @param x A real number.
#' @returns A real number.
#' @details The default test function is `f(x) = sin(x) + x`.
#' @export
test_fun <- function(x) {
  sin(x) + x
}

#' Gradient of the test function implemented to test custom gradient functionality.
#' @param x A real number.
#' @returns A real number between 0 and 2.
#' @details The gradient of default test function is `f'(x) = cos(x) + 1`.
#' @export
test_fun_grad <- function(x) {
  cos(x) + 1
}

#' Log posterior of given observations.
#' @param mu Mean of the observations.
#' @param y Vector of observations.
#' @returns A scalar of containing log posterior evaluated at its arguments.
#' @export
logpost <- function(mu, y) {
  lprior <- dunif(mu, min = 0, max = 100, log = TRUE)
  llikelihood <- sum(dnorm(y, test_fun(mu), sd=1, log=TRUE))

  lprior + llikelihood
}

#' Gradient of the log posterior of given observations.
#' @param mu Mean of the observations.
#' @param y Vector of observations.
#' @returns A scalar of containing gradient of the log posterior evaluated at its arguments.
#' @export
logpost_grad <- function(mu, y) {
  sum((y - test_fun(mu)) * (cos(mu) + 1))
}

#' Autocorrelation function of the Power Exponential Model
#' @param t Vector of time steps of the decay being modelled.
#' @param lambda Scaling factor.
#' @param rho Autocorrelation decay parameter.
#' @param sigma Variance.
#' @returns A vector of autocorrelations based on the arguments
#' @export
pex_acf <- function(t, lambda, rho, sigma) {
  sigma^2 * exp( - abs(t / lambda)^rho )
}
