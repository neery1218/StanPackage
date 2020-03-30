functions {
  real test_fun(real x); // defined in c++ file test_fun.hpp
  real test_distr_lpdf(real y, real mu); // need _lpdf (see stan reference 9.3) defined in test_fun_dist.hpp
}

data {
  int<lower=0> N;
}

parameters {
  real y;
  real mu;
}

model {
  mu ~ uniform(0, 100);
  y ~ test_distr_lpdf(mu);
}

