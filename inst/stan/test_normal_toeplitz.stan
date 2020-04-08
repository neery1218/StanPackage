functions {
  real normal_toeplitz_lpdf(real[] y, real[] acf); // need _lpdf (see stan reference 9.3) defined in test_fun_dist_vector.hpp

  // power-exponential autocorrelation function
  // used to generate a psd toeplitz matrix.
  real[] psd_acf(int N, real lambda, real rho, real sigma) {
    real acf_out[N];
    for (i in 1:N) {
      acf_out[i] = sigma^2 * exp(-fabs(i/lambda)^rho);
    }
    return acf_out;
  }

}

data {
  int<lower=0> N;
  real y[N];
}

parameters {
  real<lower=0> lambda;
  real<lower=0> rho;
  real<lower=0> sigma;
}

model {
  y ~ normal_toeplitz(psd_acf(N, lambda, rho, sigma));
}

