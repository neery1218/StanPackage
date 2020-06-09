functions {
  // mandatory declaration
  real normal_toeplitz_lpdf(real[] y, real[] acf, real[] mu);

  // acf formula
  real[] acf_fun(int N, int K, real H, real delta_t) {
    real acf_out[N * K];

    for (i in 1:(N * K)) {
      acf_out[i] = 0.5 * (delta_t / K) ^ (2*H) * (fabs(i)^(2*H) + fabs(i - 2)^(2*H) - 2 *fabs(i - 1)^(2*H));
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

  // H could be constrained to [0, 1] instead of just [0, inf],
  // but I couldn't figure out the correct constraint transform
  // for my gradient unit tests. Instead I provide a lower bound constraint and
  // add a uniform(0, 1) prior in the model block to get the same effect.
  real<lower=0> H;

  real mu;

  // TODO: can gamma be negative? Seems useless to me.
  real<lower=0> gamma;


  // TODO: add sigma back in
  real<lower=0> sigma;

  real Xt_k_fill[N * (K - 1)]; // k-level approximation filler values.
}

transformed parameters {
  // Xt U Xt_k_fill = Xt_k (length N * K)
  real Xt_k[N * K];
  real dX[N * K];
  real mus[N * K];
  real dG[N*K];

  // fill in Xt_k with observed values (data) and filler values
  Xt_k[ii_obs] = Xt;

  if (K > 1) {
    Xt_k[ii_mis] = Xt_k_fill;
  }

  // fill in dX
  dX[1] = Xt_k[1] - X0;
  for (i in 2:(N*K)) {
    dX[i] = Xt_k[i] - Xt_k[i - 1];
  }

  // calculate mus
  mus[1] = fou_mu(X0, gamma, mu);
  for (i in 2:(N*K)) {
    mus[i] = fou_mu(Xt_k[i - 1], gamma, mu);
  }

  // Xt_k[i] ~= Xt_k[i - 1] + fou_mu(Xt_k[i - 1], gamma, mu) * (delta_t / K) + sigma * dG[i]
  // => dG[i] ~= (Xt_k[i] - Xt_k[i - 1]) - fou_mu(Xt_k[i - 1], gamma, mu) * (delta_t / K) / sigma
  // can't write in one line because these are real arrays. I don't think there's any significant
  // performance hit though.
  for (i in 1:(N*K)) {
    dG[i] = (dX[i] - mus[i] * (delta_t / K)) / sigma;
  }
}

model {
  H ~ uniform(0, 1); // not completely necessary, but it makes my unit testing easier. See definition of H above.
  gamma ~ uniform(0, 2);
  mu ~ normal(0, 10); // fairly wide,for simulation purposes. i was observing the mcmc sample values like 1e9 when our N was small (~50), so this helps our mcmc converge.
  sigma ~ uniform(0, 10);

  // real normal_toeplitz(acf, mus)
  // dG ~ NormalToeplitz(0, toeplitz(acf))
  dG ~ normal_toeplitz(acf_fun(N, K, H, delta_t), rep_array(0.0, N * K));

  /*
   Stan warning: Left-hand side of sampling statement (~) may contain a non-linear transform of a parameter or local variable.
  If it does, you need to include a target += statement with the log absolute determinant of the Jacobian of the transform.
  Left-hand-side of sampling statement:
      dG ~ normal_toeplitz(...)
  */
  // included the jacobian transform, still getting the warning
  // sigma is permanently set to 1
  target += ( -1 * N * K * log(sigma) );
}
