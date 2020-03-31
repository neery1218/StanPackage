#pragma once

#include "test_fun.hpp"

template <typename T0__>
Eigen::Matrix<typename boost::math::tools::promote_args<T0__>::type, Eigen::Dynamic, 1>
test_fun(const Eigen::Matrix<T0__, Eigen::Dynamic, 1>& x, std::ostream* pstream__) {
  Eigen::Matrix<typename boost::math::tools::promote_args<T0__>::type, Eigen::Dynamic, 1> out(x.rows());
  for (int i = 0; i < x.rows(); ++i) {
    out(i, 1) = test_fun(x(i, 1), pstream__);
  }

  return out;
}

template <bool propto, typename T0__, typename T1__>
typename boost::math::tools::promote_args<T0__, T1__>::type
test_distr_lpdf(const Eigen::Matrix<T0__, Eigen::Dynamic, 1>& y,
                    const Eigen::Matrix<T1__, Eigen::Dynamic, 1>& mu, std::ostream* pstream__){
  if (y.rows() != mu.rows()) {
    stringstream errmsg;
    errmsg << "vector lengths of mu and y differ!";
    throw std::domain_error(errmsg.str());
  }
  // covariance matrix is identity matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sigma = Eigen::MatrixXd::Identity(y.rows(), y.rows());

  return stan::math::multi_normal_lpdf(y, test_fun(mu, pstream__), sigma);
}
