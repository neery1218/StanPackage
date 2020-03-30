#pragma once

#include "test_fun.hpp"

template <bool propto, typename T0__, typename T1__>
typename boost::math::tools::promote_args<T0__, T1__>::type
test_distr_lpdf(const T0__& y,
    const T1__& mu, std::ostream* pstream__)
{
  return stan::math::normal_lpdf(y, test_fun(mu, pstream__), 1);
}
