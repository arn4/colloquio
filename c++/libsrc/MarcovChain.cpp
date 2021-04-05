
#include <BinaryRBM.hpp>
using namespace rbm;

#include <cstddef>
#include <vector>
#include <random>

template<typename real_value>
const real_value MarcovChain<real_value>::default_init_probabality = real_value(0.5);

template<typename real_value>
MarcovChain<real_value>::MarcovChain(BinaryRBM<real_value>& rbm, std::mt19937& rng) :
_rbm(rbm),
_v(_rbm.m()),
_h(_rbm.n()),
_rng(rng)
{}

template<typename real_value>
void MarcovChain<real_value>::init_random_h(real_value prob_one) {
  std::binomial_distribution<int> d((unsigned int)(1), double(prob_one));
  // this is not allowed because of the strange implementation of vector of bool:
  // for(bool& hj: _h) {
  for (std::size_t j = 0; j < _h.size(); j++) {
    _h[j] = (d(_rng)==(unsigned int)(1) ? true:false);
  }
}

template<typename real_value>
void MarcovChain<real_value>::init_random_v(real_value prob_one) {
  std::binomial_distribution<int> d((unsigned int)(1), double(prob_one));
  for (std::size_t i = 0; i < _v.size(); i++) {
    _v[i] = (d(_rng)==(unsigned int)(1) ? true:false);
  }
}

template<typename real_value>
void MarcovChain<real_value>::next_step_v() {
  std::uniform_real_distribution<double> probability(real_value(0.),real_value(1.));
  for (std::size_t i = 0; i < _rbm.n(); i++) {
    if (probability(_rng) <=  _rbm.prob_v(i, _h.begin())) {
      _v[i] = true;
    } else {
      _v[i] = false;
    }
  }
}

template<typename real_value>
void MarcovChain<real_value>::next_step_h() {
  std::uniform_real_distribution<double> probability(real_value(0.),real_value(1.));
  for (std::size_t j = 0; j < _rbm.m(); j++) {
    if (probability(_rng) <= _rbm.prob_h(j, _h.begin())) {
      _h[j] = true;
    } else {
      _h[j] = false;
    }
  }
}

// Explicit Instantiation
namespace rbm {
  template class MarcovChain<float>;
  template class MarcovChain<double>;
  template class MarcovChain<long double>;
}