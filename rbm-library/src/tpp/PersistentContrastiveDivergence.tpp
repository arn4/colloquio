#include <ContrastiveDivergence.hpp>

#include <cstddef>

template<typename real_value, std::size_t features_size, std::size_t batch_size>
PersistentContrastiveDivergence<real_value, features_size, batch_size>::PersistentContrastiveDivergence(
  BernoulliRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  unsigned k,
  std::mt19937& rng,
  real_value lr,
  real_value wd,
  real_value m
) : ContrastiveDivergence<real_value, features_size, batch_size>(rbm, training_set, k, rng, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void PersistentContrastiveDivergence<real_value, features_size, batch_size>::batch_precomputing(std::size_t) {
  for (std::size_t k = 0; k < batch_size; k++) {
    PersistentContrastiveDivergence::_chains[k].evolve(PersistentContrastiveDivergence::_k);
    for (std::size_t j = 0; j < PersistentContrastiveDivergence::_rbm.n(); j++) {
      PersistentContrastiveDivergence::prob_j_table[PersistentContrastiveDivergence::_rbm.n()*k+j] = PersistentContrastiveDivergence::_rbm.prob_h(j, PersistentContrastiveDivergence::_chains[k].v().begin());
    }
  }
}