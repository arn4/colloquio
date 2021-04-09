#include <ContrastiveDivergence.hpp>
#include <TrainingAlgorithm.hpp>
#include <MarcovChain.hpp>
#include <utility.cpp>
#include <iostream>

#include <cstddef>

template<typename real_value, std::size_t features_size, std::size_t batch_size>
ContrastiveDivergence<real_value, features_size, batch_size>::ContrastiveDivergence(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  unsigned k,
  std::mt19937& rng,
  real_value lr,
  real_value wd,
  real_value m
) :
  TrainingAlgorithm<real_value, features_size, batch_size>(rbm, training_set, lr, wd, m),
  _rng(rng),
  _chains(batch_size, MarcovChain<real_value>(rbm, rng)),
  _k(k)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void ContrastiveDivergence<real_value, features_size, batch_size>::batch_precomputing(std::size_t b) {
  for (std::size_t k = 0; k < batch_size; k++) {
    auto it = ContrastiveDivergence::_training_set.batch(b).get_iterator(k);
    _chains[k].set_v(it, it + features_size);
    _chains[k].evolve(_k);
  }
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void ContrastiveDivergence<real_value, features_size, batch_size>::epoch_precomputing(std::size_t) {}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value ContrastiveDivergence<real_value, features_size, batch_size>::w_second_term(std::size_t i, std::size_t j, std::size_t k) {
  return -bool2real<real_value>(_chains[k].v()[i]) * ContrastiveDivergence::_rbm.prob_h(j, _chains[k].v().begin());
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value ContrastiveDivergence<real_value, features_size, batch_size>::b_second_term(std::size_t i, std::size_t k) {
  return -bool2real<real_value>(_chains[k].v()[i]);
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value ContrastiveDivergence<real_value, features_size, batch_size>::c_second_term(std::size_t j, std::size_t k) {
  return -ContrastiveDivergence::_rbm.prob_h(j, _chains[k].v().begin());
}