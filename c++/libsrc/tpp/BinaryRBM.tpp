#include <MarcovChain.hpp>
#include <BinaryRBM.hpp>

using namespace rbm;

#include <cstddef>
#include <vector>
#include <utility.cpp>

// Inline functions

template <typename real_value>
inline const std::size_t& BinaryRBM<real_value>::n() const {
  return _n;
}

template <typename real_value>
inline const std::size_t& BinaryRBM<real_value>::m() const {
  return _m;
}

template <typename real_value>
inline const real_value& BinaryRBM<real_value>::b(std::size_t i) const {
  return _b[i];
}

template <typename real_value>
inline const real_value& BinaryRBM<real_value>::c(std::size_t j) const {
  return _c[j];
}

template <typename real_value>
inline const real_value& BinaryRBM<real_value>::w(std::size_t i, std::size_t j) const {
  return _w[_m*i+j];
}

template <typename real_value>
inline void BinaryRBM<real_value>::update_b(std::size_t i, real_value upd) {
  _b[i] += upd;
}

template <typename real_value>
inline void BinaryRBM<real_value>::update_c(std::size_t j, real_value upd) {
  _c[j] += upd;
}

template <typename real_value>
inline void BinaryRBM<real_value>::update_w(std::size_t i, std::size_t j, real_value upd) {
  _w[_m*i+j] += upd;
}

// Template functions
template<typename real_value>
template<class Iterator>
inline real_value BinaryRBM<real_value>::prob_v(std::size_t i, Iterator h_begin) const {
  real_value sum_var = real_value(0.);
  for (std::size_t j = 0; j < _n; j++) {
    sum_var += w(i,j) * bool2real<real_value>(*h_begin);
    h_begin++;
  }
  return sigmoid(sum_var+b(i));
}

template<typename real_value>
template<class Iterator>
inline real_value BinaryRBM<real_value>::prob_h(std::size_t j, Iterator v_begin) const {
  real_value sum_var = real_value(0.);
  for (std::size_t i = 0; i < _m; i++) {
    sum_var += w(i,j) * bool2real<real_value>(*v_begin);
    v_begin++;
  }
  return sigmoid(sum_var+c(j));
}