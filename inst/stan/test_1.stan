// identical to test_1.stan, except with a constrained mu.
functions {
  real test_fun(real x); // defined in c++ file test_fun.hpp
}

data {
  int<lower=0> N;
  vector[N] y;
}

parameters {
  real<lower=0> mu;
}

model {
  mu ~ uniform(0, 100);
  y ~ normal(test_fun(mu), 1);
}
