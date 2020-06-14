functions {
  // mandatory declaration
  real normal_circulant_lpdf(real[] y, real[] acf, real[] mu);

  // acf formula
  real[] acf_fun(int N, real H, real delta_t) {
    real acf_out[N];

    for (i in 1:N) {
      acf_out[i] = 0.5 * delta_t ^ (2*H) * (fabs(i)^(2*H) + fabs(i - 2)^(2*H) - 2 *fabs(i - 1)^(2*H));
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


  real<lower=0> sigma;

  real dG_aug[N - 2];
}

transformed parameters {
  real dX[N];
  real mus[N];
  real dG[2*N - 2];

  // fill in dX
  dX[1] = Xt[1] - X0;
  for (i in 2:N) {
    dX[i] = Xt[i] - Xt[i - 1];
  }

  // calculate mus
  mus[1] = fou_mu(X0, gamma, mu);
  for (i in 2:N) {
    mus[i] = fou_mu(Xt[i - 1], gamma, mu);
  }

  // first N dGs are used, since dGs[1:N] ~ NormalToeplitz(0, toeplitz(acf))
  for (i in 1:N) {
    dG[i] = (dX[i] - mus[i] * delta_t) / sigma;
  }

  // fill in missing values for dG
  for (i in 1:(N-2)) {
    dG[N + i] = dG_aug[i];
  }
}

model {
  H ~ uniform(0, 1);
  gamma ~ uniform(0, 2);
  mu ~ normal(0, 10);
  sigma ~ uniform(0, 10);

  dG ~ normal_circulant(acf_fun(N, H, delta_t), rep_array(0.0, 2 * N - 2));
  target += ( -1 * (2*N - 2) * log(sigma) ); // FIXME: not sure if this is correct
}
