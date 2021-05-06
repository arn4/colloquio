#ifndef __PersistentContrastiveDivergence_hpp__
#define __PersistentContrastiveDivergence_hpp__

#include <ContrastiveDivergence.hpp>

#include <cstddef>
#include <vector>

namespace rbm {
  template<
    typename real_value,
    std::size_t features_size,
    std::size_t batch_size
  >
  class PersistentContrastiveDivergence: public ContrastiveDivergence<real_value, features_size, batch_size> {
    public:
      PersistentContrastiveDivergence(
        BernoulliRBM<real_value>& rbm,
        TrainingSet<features_size, batch_size>& training_set,
        unsigned k,
        std::mt19937& rng,
        real_value lr = 0.,
        real_value wd = 0.,
        real_value m = 0.
      );

      void batch_precomputing(std::size_t b);
  };
}

#include <PersistentContrastiveDivergence.tpp>
#endif