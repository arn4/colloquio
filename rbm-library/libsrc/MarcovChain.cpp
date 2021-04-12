
#include <MarcovChain.hpp>
using namespace rbm;

#include <cstddef>
#include <vector>
#include <random>
#include <iostream>

template<typename real_value>
MarcovChain<real_value>::MarcovChain(BinaryRBM<real_value>& rbm, std::mt19937& rng) :
  _rbm(rbm),
  _v(_rbm.m()),
  _h(_rbm.n()),
  _rng(rng),
  fast_rng()
{}

template<typename real_value>
void MarcovChain<real_value>::init_random_h(real_value prob_one) {
  std::binomial_distribution<binary_value> d((binary_value)(1), double(prob_one));
  for (std::size_t j = 0; j < _h.size(); j++) {
    _h[j] = d(_rng);
  }
}

template<typename real_value>
void MarcovChain<real_value>::init_random_v(real_value prob_one) {
  std::binomial_distribution<binary_value> d((binary_value)(1), double(prob_one));
  for (std::size_t i = 0; i < _v.size(); i++) {
    _v[i] = d(_rng);
  }
}

template<typename real_value>
void MarcovChain<real_value>::next_step_v() {
  std::uniform_real_distribution<real_value> probability(real_value(0.),real_value(1.));
  std::vector <real_value> prob_v = _rbm.vec_prob_v(_h.begin());
  for (std::size_t i = 0; i < _rbm.m(); i++) {
    if (probability(_rng) <=  prob_v[i]) {
      _v[i] = 1;
    } else {
      _v[i] = 0;
    }
  }
}

template<typename real_value>
void MarcovChain<real_value>::next_step_h() {
  std::uniform_real_distribution<real_value> probability(real_value(0.),real_value(1.));
  std::vector <real_value> prob_h = _rbm.vec_prob_h(_v.begin());
  for (std::size_t j = 0; j < _rbm.n(); j++) {
    if (probability(_rng) <= prob_h[j]) {
      _h[j] = 1;
    } else {
      _h[j] = 0;
    }
  }
}

// Explicit Instantiation
namespace rbm {
  template class MarcovChain<float>;
  template class MarcovChain<double>;
  template class MarcovChain<long double>;
}