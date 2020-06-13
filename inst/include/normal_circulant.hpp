#pragma once

//[[Rcpp::depends("SuperGauss")]]
#include "SuperGauss/NormalCirculant.h"
#include "helpers.hpp"
#include <iostream>
#include <vector>

template <typename Ty__, typename Tacf__, typename Tmu__>
stan::math::var normal_circulant_lpdfi(const std::vector<Ty__>& y,
    const std::vector<Tacf__>& acf, const std::vector<Tmu__>& mu, std::ostream* pstream__)
{
  using stan::math::operands_and_partials;
  using stan::math::var;

  std::vector<var> y_var = to_var(y);
  std::vector<var> acf_var = to_var(acf);
  std::vector<var> mu_var = to_var(mu);

  operands_and_partials<std::vector<var>, std::vector<var>, std::vector<var>> o(y_var, acf_var, mu_var);

  int N = y.size();

  std::vector<double> y_ = value_of(y);
  std::vector<double> acf_ = value_of(acf);
  std::vector<double> mu_ = value_of(mu);

  // NormalToeplitz solver provides an interface for solving z' ~ NormalToeplitz(0, toep)
  // We want to solve y ~ NormalToeplitz(mu, toep).
  // set z' = y_ - mu_ which has the required distribution of NormalToeplitz(0, toep)

  // calculate z'
  std::vector<double> z_(N, 0.0);
  for (int i = 0; i < y_.size(); ++i) {
    z_[i] = y_[i] - mu_[i];
  }

  // solve for log-posterior
  NormalCirculant solver(N);

  double lp = solver.logdens(z_.data(), acf_.data());

  // get gradients
  vector<double> dldz(N, 0.0);
  vector<double> dldacf(N, 0.0);

  solver.grad_full(dldz.data(),
      dldacf.data(),
      z_.data(), acf_.data(), true, true);

  // stuff into Eigen vectors (required by operands and partials)
  Eigen::VectorXd dldy_eigen(N);
  Eigen::VectorXd dldacf_eigen(N);

  for (int i = 0; i < dldz.size(); ++i) {
    dldy_eigen(i) = dldz[i]; // gradients are same
    dldacf_eigen(i) = dldacf[i];
  }

  // set partials
  // dl/dy = dl/dz' * dz'/dy = dl/dz
  o.edge1_.partials_vec_[0] += dldy_eigen;

  // dl/dacf
  o.edge2_.partials_vec_[0] += dldacf_eigen;

  // dl/dmu = dl/dz' * dz'/dmu = -dl/dz
  o.edge3_.partials_vec_[0] += (-1 * dldy_eigen);

  // build variable
  return o.build(lp);
}

template <bool propto, typename T0__, typename T1__, typename T2__>
typename boost::math::tools::promote_args<T0__, T1__, T2__>::type
normal_circulant_lpdf(const std::vector<T0__>& y,
    const std::vector<T1__>& acf, const std::vector<T2__>& mu, std::ostream* pstream__)
{
  stan::math::var v = normal_circulant_lpdfi(y, acf, mu, pstream__);
  return cast<typename boost::math::tools::promote_args<T0__, T1__, T2__>::type>(v);
}
