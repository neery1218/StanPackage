functions {
  real foo_dist_lpdfi(real y, real mu);
  // wish this worked, but it doesn't!
  // real foo_dist_lpdf(real y, real mu) {
  //   return foo_dist_lpdfi(y, mu);
  // }
}

data {
  real y_dat;
  real mu_dat;
  int<lower=1,upper=3> type;
}

parameters {
  real y;
  real mu;
}

model {
  if(type == 1) {
    // gradient wrt mu
    // y_dat ~ foo_dist(mu);
    target += foo_dist_lpdfi(y_dat, mu);
  } else if(type == 2) {
    // gradient wrt y
    // y ~ foo_dist(mu_dat);
    target += foo_dist_lpdfi(y, mu_dat);
  } else if(type == 3) {
    // gradient wrt y and mu
    // y ~ foo_dist(mu);
    target += foo_dist_lpdfi(y, mu);
  }
}
