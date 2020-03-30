class TestFun {
  public:
  double f(const double& x)
  {
    return sin(x) + x;
  }

  double df(const double& x)
  {
    return cos(x) + 1;
  }
};

TestFun inst;

double test_fun(const double& x, std::ostream* pstream__)
{
  return inst.f(x);
}

var test_fun(const var& x, std::ostream* pstream__)
{
  double a = x.val();
  double fa = sin(a) + a;
  double dfa_da = inst.df(a);
  return var(new precomp_v_vari(fa, x.vi_, dfa_da));
}

template <bool propto, typename T0__, typename T1__>
typename boost::math::tools::promote_args<T0__, T1__>::type
test_distr_lpdf(const T0__& y,
    const T1__& mu, std::ostream* pstream__)
{
  using stan::math::square; // handles both doubles and vars
  // y ~ normal(test_fun(mu), 1)
  // log pdf is -1/2 * (y - test_fun(mu))^2
  return -0.5 * square(y - test_fun(mu, pstream__));
}
