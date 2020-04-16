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

  // return dXt_{n+1}
  real fou_mu(real Xt_n, real gamma, real mu) {
    real dXt = -1 * gamma * (Xt_n - mu);
    return dXt;
  }

}

data {
  int<lower=0> N;
  int<lower=0> K;

  real X0;
  real Xt[N]; // observed data

  real<lower=0> delta_t;

  // used to index into Xt_k (See transformed parameters)
  int<lower = 1, upper = N * K> ii_obs[N];
  int<lower = 1, upper = N * K> ii_mis[N * (K - 1)];
}

parameters {

  real<lower=0, upper=1> H;
  real mu;
  real<lower=0> gamma;
  // real<lower=0> sigma; ignore for now

  real Xt_k_fill[N * (K - 1)]; // k-level approximation
  real dG[N*K];
}

transformed parameters {
  // Xt U Xt_k_fill = Xt_k (length N * K)
  real Xt_k[N * K];

  Xt_k[ii_obs] = Xt;

  if (K > 1) {
    Xt_k[ii_mis] = Xt_k_fill;
  }

  // base case
  Xt_k[1] = X0 + fou_mu(X0, gamma, mu) + delta_t / K + dG[1];

  // recursive case
  for (i in 2:(N*K)) {
    Xt_k[i] = Xt_k[i-1] + fou_mu(Xt_k[i-1], gamma, mu) * (delta_t / K) + dG[i];
  }

  // refill observed values?

}

model {
  H ~ uniform(0, 1); // stan automatically does this, i just think it's better to be explicit
  mu ~ uniform(0, 1);
  gamma ~ uniform(0, 1);

  dG ~ normal_toeplitz(acf_fun(N, K, H, delta_t), rep_array(0.0, N * K));
}
