functions {
  real normal_circulant_lpdf(real[] y, real[] acf, real[] mu); // need _lpdf (see stan reference 9.3) defined in test_fun_dist_vector.hpp

  // power-exponential autocorrelation function
  real[] psd_acf(int N, real lambda, real rho, real sigma) {
    real acf_out[N];
    for (i in 1:N) {
      acf_out[i] = sigma^2 * exp(-fabs(i/lambda)^1);
    }
    return acf_out;
  }

}

data {
  int<lower=0> N;
}

parameters {
  real y[N];

  real<lower=0> lambda;
  real<lower=0> sigma;

  real mu[N];
}

model {
  // priors
  lambda ~ uniform(0,100);
  sigma ~ uniform(0, 1);

  y ~ normal_circulant(psd_acf(N / 2 + 1, lambda, 1, sigma), mu); // implicit integer division
}
