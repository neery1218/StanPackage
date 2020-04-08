#pragma once

#include "NormalToeplitz.h"
#include <vector>

NormalToeplitz* solver = nullptr;

NormalToeplitz* get_solver(int N)
{
  // I have no idea if stan's math library is threaded. If it is, this will fail. From some basic googling, it appears that stan will _eventually_ be threaded.
  if (!solver) {
    solver = new NormalToeplitz(N);
  }
  return solver;
}

// log density
double normal_toeplitz_lpdfi(const std::vector<double>& y,
    const std::vector<double>& acf, std::ostream* pstream__)
{
  int N = y.size();
  NormalToeplitz* solver = get_solver(N);

  double lp = solver->logdens(y.data(), acf.data());
  return lp;
}

// gradient wrt y
stan::math::var normal_toeplitz_lpdfi(const std::vector<stan::math::var>& y,
    const std::vector<double>& acf, std::ostream* pstream__)
{
  int N = y.size();
  NormalToeplitz* solver = get_solver(N);

  std::vector<double> y_ = value_of(y);
  double lp = solver->logdens(y_.data(), acf.data());

  vector<double> dldy(N, 0.0);

  // TODO: if there's ever a segfault, look here.
  solver->grad_full(dldy.data(), nullptr, y_.data(), acf.data(), true, false);
  return precomputed_gradients(lp, y, dldy);
}

// gradient wrt acf
stan::math::var normal_toeplitz_lpdfi(const std::vector<double>& y,
    const std::vector<stan::math::var>& acf, std::ostream* pstream__)
{
  int N = y.size();
  NormalToeplitz* solver = get_solver(N);

  std::vector<double> acf_ = value_of(acf);
  double lp = solver->logdens(y.data(), acf_.data());

  vector<double> dlda(N, 0.0);

  // TODO: if there's ever a segfault, look here.
  solver->grad_full(nullptr, dlda.data(), y.data(), acf_.data(), false, true);
  return precomputed_gradients(lp, acf, dlda);
}

// gradient wrt y and acf
stan::math::var normal_toeplitz_lpdfi(const std::vector<stan::math::var>& y,
    const std::vector<stan::math::var>& acf, std::ostream* pstream__)
{
  int N = y.size();
  NormalToeplitz* solver = get_solver(N);

  std::vector<double> y_ = value_of(y);
  std::vector<double> acf_ = value_of(acf);

  double lp = solver->logdens(y_.data(), acf_.data());

  vector<double> dldy(N, 0.0);
  vector<double> dlda(N, 0.0);

  // TODO: if there's ever a segfault, look here.
  solver->grad_full(dldy.data(), dlda.data(), y_.data(), acf_.data(), true, true);

  /*
   * From the "Adding a new function with known gradients page." 
   * The precomputed_gradients class can be used to deal with
   * any form of inputs and outputs.All you need to do is pack
   * all the var arguments into one vector and their matching
   * gradients into another. */
  // TODO: this seems fishy.

  std::vector<stan::math::var> combined;
  std::vector<double> combined_gradients;
  for (int i = 0; i < N; ++i) {
    combined.push_back(y[i]);
    combined_gradients.push_back(dldy[i]);
  }
  for (int i = 0; i < N; ++i) {
    combined.push_back(acf[i]);
    combined_gradients.push_back(dlda[i]);
  }
  return precomputed_gradients(lp, combined, combined_gradients);
}

template <bool propto, typename T0__, typename T1__>
typename boost::math::tools::promote_args<T0__, T1__>::type
normal_toeplitz_lpdf(const std::vector<T0__>& y,
    const std::vector<T1__>& acf, std::ostream* pstream__)
{
  return normal_toeplitz_lpdfi(y, acf, pstream__);
}
