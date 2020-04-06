#ifndef testproject_foo_dist_hpp
#define testproject_foo_dist_hpp 1

#include <cmath>

namespace testproject {
  /// Log-density and gradient for the foo_dist distribution.
  ///
  /// The foo_dist distribution is defined as
  ///
  /// ```
  /// y ~ foo_dist(mu)  <=> y ~ N(sin(mu) + mu, 1).
  /// ```
  ///
  class foo_dist {
  private:
    const double LOG_2PI = 1.837877066409345483560659472811; // log(2pi)
  public:
    /// Log-density evaluation.
    ///
    /// @param[in] y Observation (scalar).
    /// @param[in] mu Sine parameter (scalar).
    /// @return The log-density evaluated at the inputs.
    double log_prob(double y, double mu) {
      double z = y - (sin(mu) + mu);
      return -.5 * (z*z + LOG_2PI);
    }
    /// Gradient of log-density with respect to `y`.
    double log_prob_dy(double y, double mu) {
      return (sin(mu) + mu) - y;
    }
    /// Gradient of log-density with respect to `mu`.
    double log_prob_dmu(double y, double mu) {
      return (y - (sin(mu) + mu)) * (cos(mu) + 1);
    }
  };
} // end namespace testproject
  
#endif // testproject_foo_dist_hpp
