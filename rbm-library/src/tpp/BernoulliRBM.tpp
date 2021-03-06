#include <BernoulliRBM.hpp>

using namespace rbm;

#include <cstddef>
#include <vector>
#include <utility.cpp>
#include <cmath>
#include <iostream>

// Inline functions

template <typename real_value>
inline const std::size_t& BernoulliRBM<real_value>::n() const {
  return _n;
}

template <typename real_value>
inline const std::size_t& BernoulliRBM<real_value>::m() const {
  return _m;
}

template <typename real_value>
inline const std::vector<real_value>& BernoulliRBM<real_value>::b() const {
  return _b;
}

template <typename real_value>
inline const std::vector<real_value>& BernoulliRBM<real_value>::c() const {
  return _c;
}

template <typename real_value>
inline const std::vector<real_value>& BernoulliRBM<real_value>::w() const {
  return _w;
}

template <typename real_value>
inline const real_value& BernoulliRBM<real_value>::b(std::size_t i) const {
  return _b[i];
}

template <typename real_value>
inline const real_value& BernoulliRBM<real_value>::c(std::size_t j) const {
  return _c[j];
}

template <typename real_value>
inline const real_value& BernoulliRBM<real_value>::w(std::size_t i, std::size_t j) const {
  return _w[_n*i+j];
}

template <typename real_value>
inline void BernoulliRBM<real_value>::update_b(std::size_t i, real_value upd) {
  _b[i] += upd;
}

template <typename real_value>
inline void BernoulliRBM<real_value>::update_c(std::size_t j, real_value upd) {
  _c[j] += upd;
}

template <typename real_value>
inline void BernoulliRBM<real_value>::update_w(std::size_t i, std::size_t j, real_value upd) {
  _w[_n*i+j] += upd;
}

// Template functions
template<typename real_value>
template<class Iterator>
inline real_value BernoulliRBM<real_value>::prob_v(std::size_t i, Iterator h_begin) const {
  real_value sum_var = real_value(0.);
  for (std::size_t j = 0; j < _n; j++) {
    sum_var += w(i,j) * binary2real<real_value>(*h_begin);
    h_begin++;
  }
  return sigmoid(sum_var+b(i));
}

template<typename real_value>
template<class Iterator>
inline real_value BernoulliRBM<real_value>::prob_h(std::size_t j, Iterator v_begin) const {
  real_value sum_var = real_value(0.);
  for (std::size_t i = 0; i < _m; i++) {
    sum_var += w(i,j) * binary2real<real_value>(*v_begin);
    v_begin++;
  }
  return sigmoid(sum_var+c(j));
}

template<typename real_value>
template<class Iterator>
inline std::vector<real_value> BernoulliRBM<real_value>::vec_prob_v(Iterator h_begin) const {
  std::vector<real_value> result(_m, real_value(0.));
  for (std::size_t j = 0; j < _n; j++) {
    real_value hj = binary2real<real_value>(*h_begin);
    h_begin++;
    for (std::size_t i = 0; i < _m; i++) {
      result[i] += w(i,j) * hj;
    }
  }
  for (std::size_t i = 0; i < _m; i++) {
    result[i] = sigmoid(result[i]+b(i));
  }
  return result;
}

template<typename real_value>
template<class Iterator>
inline std::vector<real_value> BernoulliRBM<real_value>::vec_prob_h(Iterator v_begin) const {
  std::vector<real_value> result(_n, real_value(0.));
  for (std::size_t i = 0; i < _m; i++) {
    real_value vi = binary2real<real_value>(*v_begin);
    v_begin++;
    for (std::size_t j = 0; j < _n; j++) {
      result[j] += w(i,j) * vi;
    }
  }
  for (std::size_t j = 0; j < _n; j++) {
    result[j] = sigmoid(result[j]+c(j));
  }
  return result;
}

template<typename real_value>
template<class Iterator>
inline real_value BernoulliRBM<real_value>::free_energy_v(Iterator v_begin) const {
  std::vector<real_value> x(_n,  real_value(0.));
  for (std::size_t j = 0; j < _n; j++) {
    x[j] += c(j);
    for (std::size_t i = 0; i < _m; i++) {
      x[j] += w(i,j) * binary2real<real_value>(*(v_begin+i));
    }
  }
  real_value free_energy = 0.;
  for (std::size_t i = 0; i < _m; i++) {
    free_energy -= binary2real<real_value>(*(v_begin+i)) * b(i);
  }
  for (std::size_t j = 0; j < _n; j++) {
    free_energy -= std::log(1.+std::exp(x[j]));
  }
  return free_energy;
}