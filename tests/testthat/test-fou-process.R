require(rstan)
require(mvtnorm)
require(rethinking)
require(matrixcalc)
require(numDeriv)

test_that("fOU process log density, gradients ", {

  # generate data
  delta_t <- 1
  N <- 50
  H <- 0.9
  X0 <- 3
  gamma <- 0.9
  mu <- 1
  K <- 3
  sigma <- 2

  ii_obs <- seq(K, N * K, K)
  ii_mis <- setdiff(seq(1, N * K), ii_obs)

  Xt <- csde_sim(N, delta_t, X0, list(gamma=gamma, mu=mu, H=H, sigma=sigma), fou_mu, function(theta) { theta$sigma }, fou_gamma)

  fit <- rstan::sampling(
    stanmodels$test_fou_process,
    data = list(Xt = tail(Xt, N), N = N, K = K, delta_t = delta_t, X0 = X0, ii_obs = ii_obs, ii_mis = ii_mis),
    iter = 1, chains = 1, algorithm = "Fixed_param")

  # generate values of the parameters in the model
  nsim <- 18
  Pars <- replicate(n = nsim,
                    expr = {
                      list(
                        theta = list(
                          H = runif(1, 0.6, 0.9),
                          gamma = runif(1, 0, 1),
                          mu = runif(1, 0, 1),
                          sigma = runif(1, 1, 10)
                        ),
                        Xt_k_fill = runif(N * (K - 1), -1, 1)
                      )
                    },
                    simplify = FALSE)

  # log posterior calculations in R
  lpR <- sapply(1:nsim, function(ii) {
    theta <- Pars[[ii]]$theta

    Xt_k = rep(0, N * K)
    Xt_k[ii_obs] = tail(Xt, N)
    Xt_k[ii_mis] = Pars[[ii]]$Xt_k_fill

    fou_logdens(c(X0, Xt_k), delta_t / K, theta)
  })

  # log posterior calculations from Stan
  lpStan <- sapply(1:nsim, function(ii) {
    theta <- Pars[[ii]]$theta

    upars <- rstan::unconstrain_pars(object = fit, pars = list(gamma=theta$gamma, H=theta$H, mu=theta$mu, sigma=theta$sigma, Xt_k_fill = Pars[[ii]]$Xt_k_fill))
    rstan::log_prob(object = fit,
                    upars = upars,
                    adjust_transform = FALSE)
  })

  # all differences should have the same constant
  lp_diff <- lpR - lpStan
  expect_equal(lp_diff, rep(lp_diff[1], length(lp_diff)))

  # gradient calculations in R for H, mu, gamma
  lpR_grad <- sapply(1:nsim, function(ii) {
    theta <- Pars[[ii]]$theta

    Xt_k = rep(0, N * K)
    Xt_k[ii_obs] = tail(Xt, N)
    Xt_k[ii_mis] = Pars[[ii]]$Xt_k_fill

    c(
      grad(function(x) fou_logdens(c(X0, Xt_k), delta_t / K, theta=list(gamma=theta$gamma, H=x, mu=theta$mu, sigma=theta$sigma)), x=theta$H),  # dH
      grad(function(x) fou_logdens(c(X0, Xt_k), delta_t / K, theta=list(gamma=theta$gamma, H=theta$H, mu=x, sigma=theta$sigma)), x=theta$mu),  # dmu
      grad(function(x) fou_logdens(c(X0, Xt_k), delta_t / K, theta=list(gamma=x, H=theta$H, mu=theta$mu, sigma=theta$sigma)), x=theta$gamma),  # dGamma
      grad(function(x) fou_logdens(c(X0, Xt_k), delta_t / K, theta=list(gamma=theta$gamma, H=theta$H, mu=theta$mu, sigma=x)), x=theta$sigma)  # dSigma
    )
  })

  # gradient calculations in Stan
  lpStan_grad_unconstrained <- sapply(1:nsim, function(ii) {
    theta <- Pars[[ii]]$theta
    Xt_k_fill <- Pars[[ii]]$Xt_k_fill
    upars <- rstan::unconstrain_pars(object = fit, pars = list(gamma=theta$gamma, H=theta$H, mu=theta$mu, sigma=theta$sigma, Xt_k_fill = Xt_k_fill))
    rstan::grad_log_prob(fit, upars, adjust_transform = FALSE)
  })
  ParsMat <- sapply(1:nsim, function(ii) {
    c(Pars[[ii]]$theta$H, 1, Pars[[ii]]$theta$gamma, Pars[[ii]]$theta$sigma)
  })

  # divide gradient by lambda, sigma values to get gradients on the correct scale lpStan_grad <- lpStan_grad_unconstrained / ParsMat
  lpStan_grad = lpStan_grad_unconstrained[1:4, ] / ParsMat

  # gradients should be (almost) identical
  expect_equal(lpR_grad, lpStan_grad, tolerance = 1e-6) # default tolerance (1.5e-8) causes errors
})


test_that("fOU circulant process log density, gradients ", {

  # generate data
  N <- 10
  delta_t <- 1
  X0 <- 3
  Xt <- csde_sim(N, dt=delta_t, X0=X0, list(gamma=0.9, mu=1, H=0.9, sigma=2), fou_mu, function(theta) { theta$sigma }, fou_gamma)

  # FIXME: is this correct? passing in dG_aug seems strange to me.
  csde_logdens_circulant <- function(Xt, dt, theta,
                           mu_fun, sigma_fun, gamma_fun, dG_aug) {
    dX <- diff(Xt)
    N <- length(dX)
    mu <- mu_fun(Xt[1:N], theta) # drift
    sig <- sigma_fun(theta) # diffusion
    gam <- gamma_fun(theta, dt, N) # autocorrelation
    dG <- (dX - mu * dt) / sig # noise increments
    NTz <- SuperGauss::NormalCirculant$new(2*N - 2) # instantiate NTz distribution
    ld <- NTz$logdens(c(dG, dG_aug), gam)
    ld - (2*N - 2) * log(sig) # jacobian for change-of-variables dX <-> dG
  }

  fou_logdens_circulant <- function(Xt, dt, theta, dG_aug) {
    csde_logdens_circulant(Xt, dt, theta, fou_mu, function(theta) { theta$sigma }, fou_gamma, dG_aug)
  }

  fit <- rstan::sampling(
    stanmodels$test_fou_process_circulant,
    data = list(Xt = tail(Xt, N), N = N, delta_t = delta_t, X0 = X0),
    iter = 1, chains = 1, algorithm = "Fixed_param")

  # generate values of the parameters in the model
  nsim <- 18
  Pars <- replicate(n = nsim,
                    expr = {
                      list(
                        theta = list(
                          H = runif(1, 0.6, 0.9),
                          gamma = runif(1, 0, 1),
                          mu = runif(1, 0, 1),
                          sigma = runif(1, 1, 10)
                        ),
                        dG_aug = runif(N - 2, -1, 1)
                      )
                    },
                    simplify = FALSE)

  # log posterior calculations in R
  lpR <- sapply(1:nsim, function(ii) {
    theta <- Pars[[ii]]$theta
    fou_logdens_circulant(Xt, delta_t, theta, Pars[[ii]]$dG_aug)
  })

  # log posterior calculations from Stan
  lpStan <- sapply(1:nsim, function(ii) {
    theta <- Pars[[ii]]$theta

    upars <- rstan::unconstrain_pars(object = fit, pars = list(gamma=theta$gamma, H=theta$H, mu=theta$mu, sigma=theta$sigma,  dG_aug = Pars[[ii]]$dG_aug))
    rstan::log_prob(object = fit,
                    upars = upars,
                    adjust_transform = FALSE)
  })

  # all differences should have the same constant
  lp_diff <- lpR - lpStan
  expect_equal(lp_diff, rep(lp_diff[1], length(lp_diff)), tolerance=1e-2)

  # gradient calculations in R for H, mu, gamma
  lpR_grad <- sapply(1:nsim, function(ii) {
    theta <- Pars[[ii]]$theta
    dG_aug <- Pars[[ii]]$dG_aug
    c(
      grad(function(x) fou_logdens_circulant(Xt, delta_t, theta=list(gamma=theta$gamma, H=x, mu=theta$mu, sigma=theta$sigma), dG_aug), x=theta$H),  # dH
      grad(function(x) fou_logdens_circulant(Xt, delta_t, theta=list(gamma=theta$gamma, H=theta$H, mu=x, sigma=theta$sigma), dG_aug), x=theta$mu),  # dmu
      grad(function(x) fou_logdens_circulant(Xt, delta_t, theta=list(gamma=x, H=theta$H, mu=theta$mu, sigma=theta$sigma), dG_aug), x=theta$gamma),  # dgamma
      grad(function(x) fou_logdens_circulant(Xt, delta_t, theta=list(gamma=theta$gamma, H=theta$H, mu=theta$mu, sigma=x), dG_aug), x=theta$sigma)  # dsigma
    )
  })

  # gradient calculations in Stan
  lpStan_grad_unconstrained <- sapply(1:nsim, function(ii) {
    theta <- Pars[[ii]]$theta
    dG_aug <- Pars[[ii]]$dG_aug
    upars <- rstan::unconstrain_pars(object = fit, pars = list(gamma=theta$gamma, H=theta$H, mu=theta$mu, sigma=theta$sigma, dG_aug = dG_aug))
    rstan::grad_log_prob(fit, upars, adjust_transform = FALSE)
  })
  ParsMat <- sapply(1:nsim, function(ii) {
    c(Pars[[ii]]$theta$H, 1, Pars[[ii]]$theta$gamma, Pars[[ii]]$theta$sigma)
  })

  # divide gradient by lambda, sigma values to get gradients on the correct scale lpStan_grad <- lpStan_grad_unconstrained / ParsMat
  lpStan_grad = lpStan_grad_unconstrained[1:4, ] / ParsMat

  # gradients should be (almost) identical
  expect_equal(lpR_grad, lpStan_grad, tolerance = 1e-2) # default tolerance (1.5e-8) causes errors
})
