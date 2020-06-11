#pragma once

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
