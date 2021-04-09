#ifndef __ContrastiveDivergence_hpp__
#define __ContrastiveDivergence_hpp__

#include <BinaryRBM.hpp>
#include <TrainingSet.hpp>
#include <TrainingAlgorithm.hpp>
#include <MarcovChain.hpp>

#include <cstddef>
#include <vector>

namespace rbm {
  template<
    typename real_value,
    std::size_t features_size,
    std::size_t batch_size
  >
  class ContrastiveDivergence: public TrainingAlgorithm<real_value, features_size, batch_size> {
    private:
      std::mt19937& _rng;
      std::vector<real_value> prob_j_table;
    protected:
      std::vector<MarcovChain<real_value>> _chains;
      unsigned _k;
    public:
      ContrastiveDivergence(
        BinaryRBM<real_value>& rbm,
        TrainingSet<features_size, batch_size>& training_set,
        unsigned k,
        std::mt19937& rng,
        real_value lr = 0.,
        real_value wd = 0.,
        real_value m = 0.
      );

      virtual void batch_precomputing(std::size_t b);
      void epoch_precomputing(std::size_t e);

      real_value w_second_term(std::size_t i, std::size_t j, std::size_t k);
      real_value b_second_term(std::size_t i, std::size_t k);
      real_value c_second_term(std::size_t j, std::size_t k);
  };
}

#include <ContrastiveDivergence.tpp>
#endif