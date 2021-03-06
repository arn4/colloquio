#include <BernoulliRBM.hpp>
using namespace rbm;

#include <cstddef>
#include <vector>
#include <random>
#include <cassert>

template<typename real_value>
template<class Iterator>
inline MarkovChain<real_value>::MarkovChain(BernoulliRBM<real_value>& rbm, Iterator begin, Iterator end, std::mt19937& rng) {
  MarkovChain(rbm, rng);
  set_v(begin, end);
}

template<typename real_value>
inline const std::vector<binary_value>& MarkovChain<real_value>::v() const {
  return _v;
}

template<typename real_value>
inline const std::vector<binary_value>& MarkovChain<real_value>::h() const {
  return _h;
}

template<typename real_value>
template<class Iterator>
void MarkovChain<real_value>::set_v(Iterator begin, Iterator end) {
  assert(end-begin == long(_rbm.m()));
  for (std::size_t i = 0; i < _rbm.m(); i++) {
    _v[i] = binary_value(*begin);
    begin++;
  }
}

template<typename real_value>
template<class Iterator>
void MarkovChain<real_value>::set_h(Iterator begin, Iterator end) {
  assert(end-begin == long(_rbm.n()));
  for (binary_value hj: _h) {
    hj = binary_value(*begin);
    begin++;
  }
}

template<typename real_value>
inline void MarkovChain<real_value>::evolve(unsigned times) {
  while (times--) {
    next_step_h();
    next_step_v();
  }
}