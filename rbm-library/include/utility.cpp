
#ifndef __utility4rbm_cpp__
#define __utility4rbm_cpp__

#include <cmath>


namespace rbm{
  using binary_value = int_fast8_t;

  template<typename real_value>
  inline real_value bool2real(bool b) {
    return (b ? real_value(1.):real_value(0.));
  }

  template<typename real_value>
  inline real_value binary2real(binary_value b) {
    return (binary_value(b) ? real_value(1.):real_value(0.));
  }

  template<typename real_value>
  inline real_value sigmoid(real_value x) {
    return real_value(1.)/(real_value(1.) + std::exp(real_value(-1.)*x));
  }
}

#endif