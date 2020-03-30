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

stan::math::var test_fun(const stan::math::var& x, std::ostream* pstream__)
{
  double a = x.val();
  double fa = sin(a) + a;
  double dfa_da = inst.df(a);
  return stan::math::var(new precomp_v_vari(fa, x.vi_, dfa_da));
}
