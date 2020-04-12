require(rstan)
require(numDeriv)
require(mvtnorm)

# commented out because the test takes a long time.
# test_that("test_fun unconstrained mu point estimate", {
#   options(mc.cores = parallel::detectCores()) # TODO: maybe reset after end of test?
#
#   # generate some data
#   n <- 1e5
#   mu <- runif(1, 0, 100)
#   sigma <- 1
#   y <- rnorm(n, test_fun(mu), sigma)
#   data = list(N = n, y = y)
#
#   # sample from fit, get point estimate for mu
#   fit <-  sampling(stanmodels$test_0, data = data, iter = 5e4)
#   muhat <- mean(as.data.frame(fit)$mu)
#
#   # non-deterministic check, maybe this test isn't necessary.
#   expect_equal(muhat, mu, tolerance = 1e-1)
# })

test_that("test_fun unconstrained mu log posterior", {
  # generate some data
  n <- 10
  mu <- runif(1, 0, 100)
  sigma <- 1
  y <- rnorm(n, test_fun(mu), sigma)
  data = list(N = n, y = y)

  # sample from fit
  # TODO: is there a way to not sample and still get a stanfit object?
  fit <- sampling(stanmodels$test_0, data = data,
                  iter = 1, chains = 1, algorithm = "Fixed_param")

  # generate values of the parameters in the model
  nsim <- 18
  Pars <- replicate(n = nsim,
                    expr = {
                      list(mu = runif(1, 0, 100))
                    },
                    simplify = FALSE)

  # log posterior and gradient calculations in R
  lpR <- sapply(1:nsim, function(ii) {
    mu <- Pars[[ii]]$mu
    logpost(mu, y = y)
  })
  lpR_grad <- sapply(1:nsim, function(ii) {
    mu <- Pars[[ii]]$mu
    logpost_grad(mu, y = y)
  })

  # log posterior and gradient calculations in Stan
  lpStan <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(object = fit, pars = Pars[[ii]])
    rstan::log_prob(object = fit,
                    upars = upars,
                    adjust_transform = FALSE)
  })
  lpStan_grad <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(fit, pars = Pars[[ii]])
    rstan::grad_log_prob(fit, upars, adjust_transform = TRUE)
  })

  # should return a vector of identical values.
  lp_diff <- lpR - lpStan
  expect_equal(lp_diff, rep(lp_diff[1], length(lp_diff)))

  # gradients should be (almost) identical
  expect_equal(lpR_grad, lpStan_grad, tolerance = 1e-6) # default tolerance (1.5e-8) causes errors
})

test_that("test_fun constrained mu log posterior", {
  # test_1.stan has constrained mu to be positive

  # generate some data
  n <- 10
  mu <- runif(1, 0, 100)
  sigma <- 1
  y <- rnorm(n, test_fun(mu), sigma)
  data = list(N = n, y = y)
  # sample from fit
  fit <-  sampling(stanmodels$test_1, data = data,
                   iter = 1, chains = 1, algorithm = "Fixed_param")

  # generate values of the parameters in the model
  nsim <- 18
  Pars <- replicate(n = nsim,
                    expr = {
                      list(mu = runif(1, 0, 100))
                    },
                    simplify = FALSE)

  # log posterior calculations in R
  lpR <- sapply(1:nsim, function(ii) {
    mu <- Pars[[ii]]$mu
    logpost(mu, y = y)
  })
  lpR_grad <- sapply(1:nsim, function(ii) {
    mu <- Pars[[ii]]$mu
    logpost_grad(mu, y = y)
  })

  # log posterior calculations in Stan
  lpStan <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(object = fit, pars = Pars[[ii]])
    rstan::log_prob(object = fit,
                    upars = upars,
                    adjust_transform = FALSE)
  })

  # log posterior gradient calculations in stan
  # Note that Stan samples on an "uncontrained scale", i.e., transforms
  # all +ve parameters to their logs and samples on that scale.
  # however, results are typically returned on the regular scale.
  # to fix this use the adjust_transform argument.
  lpStan_grad <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(fit, pars = Pars[[ii]])

    # adjust_transform = TRUE returns d/d nu f(exp(nu)), where nu = log(mu) (1)
    # we want d/d mu f(mu)
    # note that d/d mu nu = 1/mu => d nu = (d mu) / mu (2)
    # substituting (2) into (1) gives d/d mu f(mu) = d/d nu f(exp(nu)) / mu
    # therefore, divide grad_log_prob by mu
    rstan::grad_log_prob(fit, upars, adjust_transform = FALSE) / Pars[[ii]]$mu
  })

  # should return a vector of identical values.
  lp_diff <- lpR - lpStan
  expect_equal(lp_diff, rep(lp_diff[1], length(lp_diff)))

  # gradients should be (almost) identical
  expect_equal(lpR_grad, lpStan_grad, tolerance = 1e-6) # default tolerance (1.5e-8) causes errors
})


test_that("test_fun_dist log posterior", {
  # use a custom distribution called test_dist_lpdf in test_fun_dist.stan
  logpost <- function(mu, y) {
    lprior <- dunif(mu,
                    min = 0,
                    max = 100,
                    log = TRUE)
    llikelihood <- dnorm(y, test_fun(mu), sd = 1, log = TRUE)

    lprior + llikelihood
  }

  logpost_grad_mu <- function(mu, y) {
    (y - test_fun(mu)) * (cos(mu) + 1)
  }
  logpost_grad_y <- function(mu, y) {
    -(y - test_fun(mu))
  }

  # sample from fit
  fit <- rstan::sampling(stanmodels$test_fun_dist,
                         data = list(N = 1),
                         iter = 1, chains = 1, algorithm = "Fixed_param")

  nsim <- 18
  Pars <- replicate(n = nsim,
                    expr = {
                      list(mu = runif(1, 0, 100), y = runif(1, 0, 100))
                    },
                    simplify = FALSE)

  # log posterior and gradient calculations in R
  lpR <- sapply(1:nsim, function(ii) {
    mu <- Pars[[ii]]$mu
    y <- Pars[[ii]]$y
    logpost(mu, y)
  })

  # gradient wrt mu and y
  lpR_grad <- sapply(1:nsim, function(ii) {
    mu <- Pars[[ii]]$mu
    y <- Pars[[ii]]$y
    c(logpost_grad_y(mu, y), logpost_grad_mu(mu, y))
  })

  # log posterior and gradient calculations in Stan
  lpStan <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(object = fit, pars = Pars[[ii]])
    rstan::log_prob(object = fit,
                    upars = upars,
                    adjust_transform = TRUE)
  })
  lpStan_grad <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(fit, pars = Pars[[ii]])
    rstan::grad_log_prob(fit, upars, adjust_transform = TRUE)
  })

  # should return a vector of identical values.
  lp_diff <- lpR - lpStan
  testthat::expect_equal(lp_diff, rep(lp_diff[1], length(lp_diff)))

  # gradients should be (almost) identical
  testthat::expect_equal(lpR_grad, lpStan_grad, tolerance = 1e-6) # default tolerance (1.5e-8) causes errors
})

test_that("test_fun dist vectorized log posterior", {
  # same as previous test, but mu and y are vectors (test_fun_dist_vector.stan)
  logpost <- function(mu, y) {
    lprior <- sum(dunif(
      mu,
      min = 0,
      max = 100,
      log = TRUE
    ))
    llikelihood <- sum(dnorm(y, test_fun(mu), sd = 1, log = TRUE))
    lprior + llikelihood
  }

  # log-posterior gradient
  logpost_grad_mu <- function(mu, y) {
    (y - test_fun(mu)) * (cos(mu) + 1)
  }
  logpost_grad_y <- function(mu, y) {
    -(y - test_fun(mu))
  }

  n <- 5 # dimension of vectors

  # sample from fit
  fit <- rstan::sampling(stanmodels$test_fun_dist_vector,
                         data = list(N = n),
                         iter = 1, chains = 1, algorithm = "Fixed_param")

  nsim <- 18
  Pars <- replicate(n = nsim,
                    expr = {
                      list(mu = runif(n, 0, 100), y = runif(n, 0, 100))
                    },
                    simplify = FALSE)

  # log posterior and gradient calculations in R
  lpR <- sapply(1:nsim, function(ii) {
    mu <- Pars[[ii]]$mu
    y <- Pars[[ii]]$y
    logpost(mu, y)
  })

  lpR_grad <- sapply(1:nsim, function(ii) {
    mu <- Pars[[ii]]$mu
    y <- Pars[[ii]]$y
    c(logpost_grad_y(mu, y), logpost_grad_mu(mu, y))
  })

  # log posterior and gradient calculations in Stan
  lpStan <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(object = fit, pars = Pars[[ii]])
    rstan::log_prob(object = fit,
                    upars = upars,
                    adjust_transform = TRUE)
  })
  lpStan_grad <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(fit, pars = Pars[[ii]])
    rstan::grad_log_prob(fit, upars, adjust_transform = TRUE)
  })

  # should return a vector of identical values.
  lp_diff <- lpR - lpStan
  testthat::expect_equal(lp_diff, rep(lp_diff[1], length(lp_diff)))

  # gradients should be (almost) identical
  testthat::expect_equal(lpR_grad, lpStan_grad, tolerance = 1e-6) # default tolerance (1.5e-8) causes errors
})

test_that("test_fun foo_dist all gradients", {
  foo_dist <- function(y, mu) {
    dnorm(y, sin(mu) + mu, sd = 1, log = TRUE)
  }

  y_dat <- rnorm(1)
  mu_dat <- rnorm(1)

  #--- gradient wrt mu -----------------------------------------------------------

  fit <- sampling(stanmodels$foo_dist,
                  data = list(y_dat = y_dat, mu_dat = mu_dat, type = 1),
                  iter = 1, chains = 1, algorithm = "Fixed_param")


  nsim <- 20
  Pars <- replicate(n = nsim, expr = {
    list(y = rnorm(1), mu = rnorm(1))
  }, simplify = FALSE)

  # R calcs
  lpR <- sapply(1:nsim, function(ii) {
    y <- Pars[[ii]]$y
    mu <- Pars[[ii]]$mu
    foo_dist(y_dat, mu)
  })
  lpR_grad <- sapply(1:nsim, function(ii) {
    y <- Pars[[ii]]$y
    mu <- Pars[[ii]]$mu
    grad(function(mu_) foo_dist(y_dat, mu_), x = mu)[1]
  })

  # Stan calcs
  lpStan <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(object = fit, pars = Pars[[ii]])
    log_prob(object = fit,
             upars = upars,
             adjust_transform = FALSE)
  })
  lpStan_grad <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(fit, pars = Pars[[ii]])
    rstan::grad_log_prob(fit, upars, adjust_transform = FALSE)[2]
  })

  lpR - lpStan
  lpR_grad - lpStan_grad

  #--- gradient wrt y -----------------------------------------------------------

  fit <- sampling(stanmodels$foo_dist,
                  data = list(y_dat = y_dat, mu_dat = mu_dat, type = 2),
                  iter = 1, chains = 1, algorithm = "Fixed_param")


  nsim <- 20
  Pars <- replicate(n = nsim, expr = {
    list(y = rnorm(1), mu = rnorm(1))
  }, simplify = FALSE)

  # R calcs
  lpR <- sapply(1:nsim, function(ii) {
    y <- Pars[[ii]]$y
    mu <- Pars[[ii]]$mu
    foo_dist(y, mu_dat)
  })
  lpR_grad <- sapply(1:nsim, function(ii) {
    y <- Pars[[ii]]$y
    mu <- Pars[[ii]]$mu
    grad(function(y_) foo_dist(y_, mu_dat), x = y)[1]
  })

  # Stan calcs
  lpStan <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(object = fit, pars = Pars[[ii]])
    log_prob(object = fit,
             upars = upars,
             adjust_transform = FALSE)
  })
  lpStan_grad <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(fit, pars = Pars[[ii]])
    rstan::grad_log_prob(fit, upars, adjust_transform = FALSE)[1]
  })

  lpR - lpStan
  lpR_grad - lpStan_grad

  #--- gradient wrt y and mu -----------------------------------------------------

  fit <- sampling(stanmodels$foo_dist,
                  data = list(y_dat = y_dat, mu_dat = mu_dat, type = 3),
                  iter = 1, chains = 1, algorithm = "Fixed_param")


  nsim <- 20
  Pars <- replicate(n = nsim, expr = {
    list(y = rnorm(1), mu = rnorm(1))
  }, simplify = FALSE)

  # R calcs
  lpR <- sapply(1:nsim, function(ii) {
    y <- Pars[[ii]]$y
    mu <- Pars[[ii]]$mu
    foo_dist(y, mu)
  })
  lpR_grad <- sapply(1:nsim, function(ii) {
    y <- Pars[[ii]]$y
    mu <- Pars[[ii]]$mu
    grad(function(x) foo_dist(x[1], x[2]), x = c(y, mu))
  })

  # Stan calcs
  lpStan <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(object = fit, pars = Pars[[ii]])
    rstan::log_prob(object = fit,
                    upars = upars,
                    adjust_transform = FALSE)
  })
  lpStan_grad <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(fit, pars = Pars[[ii]])
    rstan::grad_log_prob(fit, upars, adjust_transform = FALSE)
  })

  expect_equal(lpR, lpStan)
  expect_equal(lpR_grad, lpStan_grad)
})

# TODO: once a mean parameter is added, add those gradients too
test_that("normal_toeplitz log density, gradients wrt toeplitz params", {
  # generate data
  N <- 3
  max_lambda <- 100
  max_sigma <- 1
  lambda <- runif(1, 0, max_lambda)
  rho <- 1
  sigma <- runif(1, 0, max_sigma)
  toep <- toeplitz(pex_acf(1:N, lambda, 1, sigma))
  y <- rmvnorm(1,rep(0, N), toep)[1,]
  data = list(N=N, y_dat=y, type=1, lambda_dat=lambda, sigma_dat=sigma) # gradient wrt lambda, sigma

  # log posterior function
  toep_logpost <- function(y, lambda, sigma) {
    lprior <- dunif(lambda, min = 0, max = max_lambda, log = TRUE) +
      dunif(max_sigma, min=0, max=1, log=TRUE)

    N <- length(y)
    llikelihood <- dmvnorm(y, mean = rep(0, N), sigma = toeplitz(pex_acf(1:N, lambda, 1, sigma)), log = TRUE)
    lprior + llikelihood
  }

  fit <- rstan::sampling(stanmodels$test_normal_toeplitz, data = data,
                         iter = 1, chains = 1, algorithm = "Fixed_param")

  # generate values of the parameters in the model
  nsim <- 18
  Pars <- replicate(n = nsim,
                    expr = {
                      list(
                        lambda = runif(1, 0, max_lambda),
                        sigma = runif(1, 0, max_sigma),
                        y = y
                        )
                    },
                    simplify = FALSE)

  # log posterior calculations in R
  lpR <- sapply(1:nsim, function(ii) {
    lambda <- Pars[[ii]]$lambda
    sigma <- Pars[[ii]]$sigma
    toep_logpost(y, lambda, sigma)
  })
  # log posterior calculations in Stan
  lpStan <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(object = fit, pars = Pars[[ii]])
    rstan::log_prob(object = fit,
                    upars = upars,
                    adjust_transform = FALSE)
  })

  # differences should be identical
  lp_diff <- lpR - lpStan
  expect_equal(lp_diff, rep(lp_diff[1], length(lp_diff)))

  # gradients wrt lambda, sigma
  lpR_grad <- sapply(1:nsim, function(ii) {
    lambda <- Pars[[ii]]$lambda
    sigma <- Pars[[ii]]$sigma
    c(grad(function(x) toep_logpost(y, x, sigma),  x=lambda)[1], grad(function(x) toep_logpost(y, lambda, x),  x=sigma)[1])
  })

  # stan gives gradients on unconstrained scale, so we divide by ParsMat to get the gradients we care about.
  # the math for this is explained in the "constrained mu" test above.
  lpStan_grad_constrained <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(fit, pars = Pars[[ii]])
    rstan::grad_log_prob(fit, upars, adjust_transform = FALSE)
  })
  ParsMat <- sapply(1:nsim, function(ii) {
    c(Pars[[ii]]$y, Pars[[ii]]$lambda, Pars[[ii]]$sigma)
  })

  # divide gradient by lambda, sigma values to get gradients on the correct scale
  lpStan_grad <- lpStan_grad_constrained / ParsMat

  # gradients should be (almost) identical
  # expect_equal(lpR_grad, lpStan_grad, tolerance = 1e-6) # default tolerance (1.5e-8) causes errors
  expect_equal(lpR_grad, lpStan_grad[4:5,]) # default tolerance (1.5e-8) causes errors
})

test_that("normal_toeplitz log density, gradients wrt y", {
  # generate data
  N <- 3
  max_lambda <- 100
  max_sigma <- 1
  lambda <- runif(1, 0, max_lambda)
  rho <- 1
  sigma <- runif(1, 0, max_sigma)
  toep <- toeplitz(pex_acf(1:N, lambda, 1, sigma))
  y <- rmvnorm(1,rep(0, N), toep)[1,]
  data = list(N=N, y_dat=y, type=2, lambda_dat=lambda, sigma_dat=sigma) # gradient wrt lambda, sigma

  # log posterior function
  toep_logpost <- function(y, lambda, sigma) {
    lprior <- dunif(lambda, min = 0, max = max_lambda, log = TRUE) +
      dunif(max_sigma, min=0, max=1, log=TRUE)

    N <- length(y)
    llikelihood <- dmvnorm(y, mean = rep(0, N), sigma = toeplitz(pex_acf(1:N, lambda, 1, sigma)), log = TRUE)
    lprior + llikelihood
  }

  fit <- rstan::sampling(stanmodels$test_normal_toeplitz, data = data,
                         iter = 1, chains = 1, algorithm = "Fixed_param")

  # generate values of the parameters in the model
  nsim <- 18
  Pars <- replicate(n = nsim,
                    expr = {
                      list(
                        lambda = lambda,
                        sigma = sigma,
                        y = rmvnorm(1, rep(0, N), toep)[1,]
                        )
                    },
                    simplify = FALSE)

  # log posterior calculations in R
  lpR <- sapply(1:nsim, function(ii) {
    y <- Pars[[ii]]$y
    toep_logpost(y, lambda, sigma)
  })
  # log posterior calculations in Stan
  lpStan <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(object = fit, pars = Pars[[ii]])
    rstan::log_prob(object = fit,
                    upars = upars,
                    adjust_transform = FALSE)
  })

  # differences should be identical
  lp_diff <- lpR - lpStan
  expect_equal(lp_diff, rep(lp_diff[1], length(lp_diff)))

  # gradients wrt y
  lpR_grad <- sapply(1:nsim, function(ii) {
    y <- Pars[[ii]]$y
    grad(function(x) toep_logpost(x, lambda, sigma),  x=y)
  })
  lpStan_grad <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(fit, pars = Pars[[ii]])
    rstan::grad_log_prob(fit, upars, adjust_transform = FALSE)
  })

  # gradients should be (almost) identical
  # expect_equal(lpR_grad, lpStan_grad, tolerance = 1e-6) # default tolerance (1.5e-8) causes errors
  expect_equal(lpR_grad, lpStan_grad[1:3,]) # default tolerance (1.5e-8) causes errors
})

test_that("normal_toeplitz log density, gradients wrt everything", {
  # generate data
  N <- 3
  max_lambda <- 100
  max_sigma <- 1
  lambda <- runif(1, 0, max_lambda)
  rho <- 1
  sigma <- runif(1, 0, max_sigma)
  toep <- toeplitz(pex_acf(1:N, lambda, 1, sigma))
  y <- rmvnorm(1,rep(0, N), toep)[1,]
  data = list(N=N, y_dat=y, type=3, lambda_dat=lambda, sigma_dat=sigma) # gradient wrt lambda, sigma

  # log posterior function
  toep_logpost <- function(y, lambda, sigma) {
    lprior <- dunif(lambda, min = 0, max = max_lambda, log = TRUE) +
      dunif(max_sigma, min=0, max=1, log=TRUE)

    N <- length(y)
    llikelihood <- dmvnorm(y, mean = rep(0, N), sigma = toeplitz(pex_acf(1:N, lambda, 1, sigma)), log = TRUE)
    lprior + llikelihood
  }

  fit <- rstan::sampling(stanmodels$test_normal_toeplitz, data = data,
                         iter = 1, chains = 1, algorithm = "Fixed_param")

  # generate values of the parameters in the model
  nsim <- 18
  Pars <- replicate(n = nsim,
                    expr = {
                      list(
                        lambda = runif(1, 0, max_lambda),
                        sigma = runif(1, 0, max_sigma),
                        y = rmvnorm(1, rep(0, N), toep)[1,]
                        )
                    },
                    simplify = FALSE)

  # log posterior calculations in R
  lpR <- sapply(1:nsim, function(ii) {
    y <- Pars[[ii]]$y
    lambda <- Pars[[ii]]$lambda
    sigma <- Pars[[ii]]$sigma
    toep_logpost(y, lambda, sigma)
  })
  # log posterior calculations in Stan
  lpStan <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(object = fit, pars = Pars[[ii]])
    rstan::log_prob(object = fit,
                    upars = upars,
                    adjust_transform = FALSE)
  })

  # differences should be identical
  lp_diff <- lpR - lpStan
  expect_equal(lp_diff, rep(lp_diff[1], length(lp_diff)))

  # gradients wrt lambda, sigma
  lpR_grad <- sapply(1:nsim, function(ii) {
    y <- Pars[[ii]]$y
    lambda <- Pars[[ii]]$lambda
    sigma <- Pars[[ii]]$sigma
    c(
      grad(function(x) toep_logpost(x, lambda, sigma),  x=y),
      grad(function(x) toep_logpost(y, x, sigma),  x=lambda)[1],
      grad(function(x) toep_logpost(y, lambda, x),  x=sigma)[1]
    )
  })

  # stan gives gradients on unconstrained scale, so we divide by ParsMat to get the gradients we care about.
  # the math for this is explained in the "constrained mu" test above.
  lpStan_grad_constrained <- sapply(1:nsim, function(ii) {
    upars <- rstan::unconstrain_pars(fit, pars = Pars[[ii]])
    rstan::grad_log_prob(fit, upars, adjust_transform = FALSE)
  })
  ParsMat <- sapply(1:nsim, function(ii) {
    # divide y by one because it's not constrained
    c(rep(1, length(y)), Pars[[ii]]$lambda, Pars[[ii]]$sigma)
  })

  # divide gradient by lambda, sigma values to get gradients on the correct scale
  lpStan_grad <- lpStan_grad_constrained / ParsMat

  # gradients should be (almost) identical
  # expect_equal(lpR_grad, lpStan_grad, tolerance = 1e-6) # default tolerance (1.5e-8) causes errors
  expect_equal(lpR_grad, lpStan_grad) # default tolerance (1.5e-8) causes errors
})
