functions {
  // forward declaration required by rstan::stanc (tested)
  real foo_dist_lpdf(real y, real mu); 
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
    y_dat ~ foo_dist(mu);
  } else if(type == 2) {
    // gradient wrt y
    y ~ foo_dist(mu_dat);
  } else if(type == 3) {
    // gradient wrt y and mu
    y ~ foo_dist(mu);
  }
}
