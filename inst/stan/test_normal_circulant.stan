functions {
  real normal_circulant_lpdf(real[] y, real[] acf, real[] mu); // need _lpdf (see stan reference 9.3) defined in test_fun_dist_vector.hpp

  // power-exponential autocorrelation function
  // used to generate a psd toeplitz matrix.
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

  real y_dat[N];

  real<lower=0> lambda_dat;
  real<lower=0> sigma_dat;

  real mu_dat[N];

  int<lower=1,upper=4> type;
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

  if(type == 1) {
    // gradient wrt lambda, sigma
    y_dat ~ normal_circulant(psd_acf(N, lambda, 1, sigma), mu_dat);
  } else if(type == 2) {
    // gradient wrt y
    y ~ normal_circulant(psd_acf(N, lambda_dat, 1, sigma_dat), mu_dat);
  }
  else if(type == 4) {
    // gradient wrt y and lambda,sigma
    y ~ normal_circulant(psd_acf(N, lambda, 1, sigma), mu);
  }
}
