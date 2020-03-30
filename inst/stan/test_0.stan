// use analytic gradient of test_fun to sample from the posterior.
functions {
  real test_fun(real x); // defined in c++ file test_fun.hpp
}

data {
  int<lower=0> N;
  vector[N] y;
}

parameters {
  real mu;
}

model {
  mu ~ uniform(0, 100);
  y ~ normal(test_fun(mu), 1);
}
