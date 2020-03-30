# Model
# likelihood: y ~ normal(test_fun(mu), 1)
# prior: mu ~ unif(0, 100)

test_fun <- function(x) {
  sin(x) + x
}

test_fun_grad <- function(x) {
  cos(x) + 1
}

# log-posterior calculation
logpost <- function(mu, y) {
  lprior <- dunif(mu, min = 0, max = 100, log = TRUE)
  llikelihood <- sum(dnorm(y, test_fun(mu), sd=1, log=TRUE))

  lprior + llikelihood
}

# log-posterior gradient
logpost_grad <- function(mu, y) {
  # TODO: This sum _should_ be negated, but the output matches stan...
  sum((y - test_fun(mu)) * (cos(mu) + 1))
}

# accessor functions for test cases
sample_from_test_0 <- function(data, iter) {
  sampling(stanmodels$test_0, data=data, iter = iter)
}
sample_from_test_1 <- function(data, iter) {
  sampling(stanmodels$test_1, data=data, iter = iter)
}

