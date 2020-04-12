#pragma once

//[[Rcpp::depends("SuperGauss")]]
#include "NormalToeplitz.h"
#include <iostream>
#include <vector>

template <typename Ty__, typename Tacf__>
stan::math::var normal_toeplitz_lpdfi(const std::vector<Ty__>& y,
    const std::vector<Tacf__>& acf, std::ostream* pstream__)
{
  using stan::math::operands_and_partials;
  using stan::math::var;

  std::vector<var> y_var = to_var(y);
  std::vector<var> acf_var = to_var(acf);

  operands_and_partials<std::vector<var>, std::vector<var>> o(y_var, acf_var);

  int N = y.size();

  std::vector<double> y_ = value_of(y);
  std::vector<double> acf_ = value_of(acf);

  // solve for log-posterior
  NormalToeplitz solver(N);
  double lp = solver.logdens(y_.data(), acf_.data());

  // get gradients
  vector<double> dldy(N, 0.0);
  vector<double> dldacf(N, 0.0);

  solver.grad_full(dldy.data(),
      dldacf.data(),
      y_.data(), acf_.data(), true, true);

  // stuff into Eigen vectors
  Eigen::VectorXd dldy_eigen(N);
  Eigen::VectorXd dldacf_eigen(N);

  for (int i = 0; i < dldy.size(); ++i) {
    dldy_eigen(i) = dldy[i];
    dldacf_eigen(i) = dldacf[i];
  }

  // set partials
  o.edge1_.partials_vec_[0] += dldy_eigen;
  o.edge2_.partials_vec_[0] += dldacf_eigen;

  // build variable
  return o.build(lp);
}

// TODO: find the correct cast function from stan/math
template <class T>
T cast(stan::math::var v);

template <>
stan::math::var cast(stan::math::var v)
{
  return v;
}

template <>
double cast(stan::math::var v)
{
  double val = v.val();
  return val;
}

template <bool propto, typename T0__, typename T1__>
typename boost::math::tools::promote_args<T0__, T1__>::type
normal_toeplitz_lpdf(const std::vector<T0__>& y,
    const std::vector<T1__>& acf, std::ostream* pstream__)
{
  stan::math::var v = normal_toeplitz_lpdfi(y, acf, pstream__);
  return cast<typename boost::math::tools::promote_args<T0__, T1__>::type>(v);
}
