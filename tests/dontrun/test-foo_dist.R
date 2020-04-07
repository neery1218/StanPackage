# tests for foo_dist distribution

require(testproject1)
require(rstan)
require(numDeriv)

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
  upars <- unconstrain_pars(object = fit, pars = Pars[[ii]])
  log_prob(object = fit,
           upars = upars,
           adjust_transform = FALSE)
})
lpStan_grad <- sapply(1:nsim, function(ii) {
  upars <- unconstrain_pars(fit, pars = Pars[[ii]])
  grad_log_prob(fit, upars, adjust_transform = FALSE)[2]
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
  upars <- unconstrain_pars(object = fit, pars = Pars[[ii]])
  log_prob(object = fit,
           upars = upars,
           adjust_transform = FALSE)
})
lpStan_grad <- sapply(1:nsim, function(ii) {
  upars <- unconstrain_pars(fit, pars = Pars[[ii]])
  grad_log_prob(fit, upars, adjust_transform = FALSE)[1]
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
  upars <- unconstrain_pars(object = fit, pars = Pars[[ii]])
  log_prob(object = fit,
           upars = upars,
           adjust_transform = FALSE)
})
lpStan_grad <- sapply(1:nsim, function(ii) {
  upars <- unconstrain_pars(fit, pars = Pars[[ii]])
  grad_log_prob(fit, upars, adjust_transform = FALSE)
})

lpR - lpStan
lpR_grad - lpStan_grad
