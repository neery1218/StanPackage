functions {
  real test_fun(real x); // defined in c++ file test_fun.hpp
  real test_distr_lpdf(vector y, vector mu); // need _lpdf (see stan reference 9.3)
}

data {
  int<lower=0> N;
}

parameters {
  vector[N] y;
  vector[N] mu;
}

model {
  mu ~ uniform(0, 100);
  y ~ test_distr_lpdf(mu);
}

