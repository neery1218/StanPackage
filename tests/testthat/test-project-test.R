test_that("test_fun constrained mu", {
  # options(mc.cores = parallel::detectCores()) # TODO: maybe reset after end of test?

  # generate some data
  n <- 1e5
  mu <- runif(1, 0, 100)
  sigma <- 1
  y <- rnorm(n, test_fun(mu), 1)
  data = list(N = n, y = y)

  # sample from fit, get point estimate for mu
  fit <- sample_from_test_1(data, 5e4)
  muhat <- mean(as.data.frame(fit)$mu)

  # # FIXME: arbitrary tolerance for non-deterministic check...maybe remove this check?
  expect_equal(muhat, mu, tolerance = 1e-2)

  # check log-posterior ------------------------------------------

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
    rstan::grad_log_prob(fit, upars, adjust_transform = TRUE) / Pars[[ii]]$mu
  })

  # should return a vector of identical values.
  lp_diff <- lpR - lpStan
  expect_equal(lp_diff, rep(lp_diff[1], length(lp_diff)))

  # gradients should be (almost) identical
  expect_equal(lpR_grad, lpStan_grad, tolerance = 1e-6) # default tolerance (1.5e-8) causes errors
})
