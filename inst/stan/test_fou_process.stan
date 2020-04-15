functions {
  // mandatory declaration
  real normal_toeplitz_lpdf(real[] y, real[] acf, real[] mu);

  // acf formula
  real[] acf_fun(int N, int K, real H, real delta_t) {
    real acf_out[N * K];

    for (i in 1:(N * K)) {
      acf_out[i] = 0.5 * (delta_t / K) ^ (2*H) * (fabs(i + 1)^(2*H) + fabs(i - 1)^(2*H) - 2 *fabs(i)^(2*H));
    }

    return acf_out;
  }

  // given Y_N, delta_t, mu, return Y_N+1
  // dY_t = -gamma * (Y_t - mu)dt
  real mu_fun_fou(real y_n, real delta_t, real gamma, real mu) {
    real delta_y = -1 * gamma * (y_n - mu) * delta_t;
    return delta_y;
  }

  // apply mu_fun_fou to get the mean of the Gaussian process
  // TODO: does stan support functions as arguments? I think no...
  real[] get_mus(real delta_t, int N, int K, real gamma, real mu, real Y_0) {
    real mu_out[N * K];

    // base case
    mu_out[1] = Y_0 + mu_fun_fou(Y_0, delta_t / K, gamma, mu);

    // recursive case
    for (i in 2:(N * K)) {
      mu_out[i] = mu_out[i - 1] + mu_fun_fou(mu_out[i-1], delta_t / K, gamma, mu);
    }

    return mu_out;
  }

}

data {
  int<lower=0> N;
  real y_0;
  real y_obs[N];

  real<lower=0> delta_t;
  int<lower=0> K;
}

parameters {
  // y_obs U y_fill = y_complete
  real y_fill[N * (K - 1)];

  real<lower=0, upper=1> H;
  real mu;
  real gamma;
  // real sigma; ignore for now
}

model {
  real y_complete[N * K];

  int obs_index = 1;
  int fill_index = 1;

  // fill in y_complete
  for (i in 1:(N * K)) {
    if (i % K == 0) {
      y_complete[i] = y_obs[obs_index];
      obs_index = obs_index + 1;
    } else {
      y_complete[i] = y_fill[fill_index];
      fill_index = fill_index + 1;
    }
  }

  y_complete ~ normal_toeplitz(
    acf_fun(N, K, H, delta_t),
    get_mus(delta_t, N, K, gamma, mu, y_0));
}
