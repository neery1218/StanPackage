#ifndef testproject_foo_dist_stan_hpp
#define testproject_foo_dist_stan_hpp 1

#include "foo_dist.hpp"
#include <stan/math/rev/core.hpp>

// plain function
// NOTE: for some reason doesn't work with _lpdf!
double foo_dist_lpdfi(const double& y, const double& mu,
		     std::ostream* pstream__) {
  testproject::foo_dist fd;
  return fd.log_prob(y, mu);
}

// gradient wrt y
stan::math::var foo_dist_lpdfi(const stan::math::var& y, const double& mu,
			      std::ostream* pstream__) {
  testproject::foo_dist fd;
  double y_ = y.val();
  double lp = fd.log_prob(y_, mu);
  double lp_dy = fd.log_prob_dy(y_, mu);
  return stan::math::var(new precomp_v_vari(lp, y.vi_, lp_dy));
}

// gradient wrt mu
stan::math::var foo_dist_lpdfi(double y, const stan::math::var& mu,
			      std::ostream* pstream__) {
  testproject::foo_dist fd;
  double mu_ = mu.val();
  double lp = fd.log_prob(y, mu_);
  double lp_dmu = fd.log_prob_dmu(y, mu_);
  return stan::math::var(new precomp_v_vari(lp, mu.vi_, lp_dmu));
}

// gradient wrt y and mu
stan::math::var foo_dist_lpdfi(const stan::math::var& y,
			      const stan::math::var& mu,
			      std::ostream* pstream__) {
  testproject::foo_dist fd;
  double y_ = y.val();
  double mu_ = mu.val();
  double lp = fd.log_prob(y_, mu_);
  double lp_dy = fd.log_prob_dy(y_, mu_);
  double lp_dmu = fd.log_prob_dmu(y_, mu_);
  return stan::math::var(new precomp_vv_vari(lp, y.vi_, mu.vi_, lp_dy, lp_dmu));
}

#endif // testproject_foo_dist_stan_hpp
