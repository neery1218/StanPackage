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
  real X0;
  real Xt[N]; // observed data

  real<lower=0> delta_t;
  int<lower=0> K;
}

parameters {

  real<lower=0, upper=1> H;
  real mu;
  real gamma;
  // real<lower=0> sigma; ignore for now

  // Xt U Xt_k_fill = Xt_k (length N * K)
  real Xt_k_fill[N * (K - 1)]; // k-level approximation
  real dG[N*K];
}

model {
  vector[N*K] Xt_k;

  real zeroes[N*K];
  int obs_index = 1;
  int fill_index = 1;

  // it's hard to justify using these priors
  H ~ uniform(0, 1);
  mu ~ uniform(-5, 5);
  gamma ~ uniform(0, 10);

  // fill in Xt_k using observed values (Xt) and filler values (Xt_k_fill)
  for (i in 1:(N * K)) {
    if (i % K == 0) {
      Xt_k[i] = Xt[obs_index];
      obs_index = obs_index + 1;
    } else {
      Xt_k[i] = Xt_k_fill[fill_index];
      fill_index = fill_index + 1;
    }
  }

  // base case
  Xt_k[1] = X0 + fou_mu(X0, gamma, mu) + delta_t / K + dG[1];
  for (i in 2:(N*K)) {
    Xt_k[i] = Xt_k[i-1] + fou_mu(Xt_k[i-1], gamma, mu) * (delta_t / K) + dG[i];
  }

  for (i in 1:(N*K)) {
    zeroes[i] = 0;
  }

  dG ~ normal_toeplitz(acf_fun(N, K, H, delta_t), zeroes);
}
