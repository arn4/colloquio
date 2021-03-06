#include <ContrastiveDivergence.hpp>
#include <TrainingAlgorithm.hpp>
#include <MarkovChain.hpp>
#include <utility.cpp>
#include <iostream>

#include <cstddef>

template<typename real_value, std::size_t features_size, std::size_t batch_size>
ContrastiveDivergence<real_value, features_size, batch_size>::ContrastiveDivergence(
  BernoulliRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  unsigned k,
  std::mt19937& rng,
  real_value lr,
  real_value wd,
  real_value m
) :
  TrainingAlgorithm<real_value, features_size, batch_size>(rbm, training_set, lr, wd, m),
  _rng(rng),
  prob_j_table(rbm.n()*batch_size),
  _chains(batch_size, MarkovChain<real_value>(rbm, rng)),
  _k(k)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void ContrastiveDivergence<real_value, features_size, batch_size>::batch_precomputing(std::size_t b) {
  for (std::size_t k = 0; k < batch_size; k++) {
    auto it = ContrastiveDivergence::_training_set.batch(b).get_iterator(k);
    _chains[k].set_v(it, it + features_size);
    _chains[k].evolve(_k);
    for (std::size_t j = 0; j < ContrastiveDivergence::_rbm.n(); j++) {
      ContrastiveDivergence::prob_j_table[ContrastiveDivergence::_rbm.n()*k+j] = ContrastiveDivergence::_rbm.prob_h(j, ContrastiveDivergence::_chains[k].v().begin());
    }
  }
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void ContrastiveDivergence<real_value, features_size, batch_size>::epoch_precomputing(std::size_t) {}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value ContrastiveDivergence<real_value, features_size, batch_size>::w_second_term(std::size_t i, std::size_t j, std::size_t k) {
  return -bool2real<real_value>(_chains[k].v()[i]) * prob_j_table[ContrastiveDivergence::_rbm.n()*k+j];
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value ContrastiveDivergence<real_value, features_size, batch_size>::b_second_term(std::size_t i, std::size_t k) {
  return -bool2real<real_value>(_chains[k].v()[i]);
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value ContrastiveDivergence<real_value, features_size, batch_size>::c_second_term(std::size_t j, std::size_t k) {
  return -prob_j_table[ContrastiveDivergence::_rbm.n()*k+j];
}