functions {
  vector test_fun(vector x); // defined in c++ file test_fun.hpp
  real test_distr_lpdf(vector y, vector mu); // need _lpdf (see stan reference 9.3) defined in test_fun_dist_vector.hpp
}

data {
  int<lower=0> N;
}

parameters {
  vector[N] y;
  vector[N] mu;
}

model {
  // TODO: is mu ~ uniform(0, 100) enough?
  for (i in 1:N) {
    mu[i] ~ uniform(0, 100);
  }
  y ~ test_distr_lpdf(mu);
}

