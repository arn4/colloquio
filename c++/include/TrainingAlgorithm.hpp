#ifndef __TrainingAlgorithm_hpp__
#define __TrainingAlgorithm_hpp__

#include <BinaryRBM.hpp>
#include <TrainingSet.hpp>
#include <cstddef>
#include <vector>

/* This is a base class for other training algorithm,
 * it's not usable alone.
 */

namespace rbm {
  template<
    typename real_value,
    std::size_t features_size,
    std::size_t batch_size
  >
  class TrainingAlgorithm {
    private:
      std::vector<real_value> _last_update_b;
      std::vector<real_value> _last_update_c;
      std::vector<real_value> _last_update_w;

      void train_on_batch(std::size_t b);

    protected:
      BinaryRBM<real_value>& _rbm;
      TrainingSet<features_size, batch_size>& _training_set;
    
    public:
      real_value learning_rate, weight_decay, momentum;

      TrainingAlgorithm(
        BinaryRBM<real_value>& rbm,
        TrainingSet<features_size, batch_size>& training_set,
        real_value lr = 0.,
        real_value wd = 0.,
        real_value m = 0.
      );

      // Methods to be overriden in specialized training algorithms
      virtual void batch_precomputing(std::size_t b) = 0;
      virtual void epoch_precomputing() = 0;

      virtual real_value w_second_term(std::size_t i, std::size_t j, std::size_t k) = 0;
      virtual real_value b_second_term(std::size_t i, std::size_t k) = 0;
      virtual real_value c_second_term(std::size_t j, std::size_t k) = 0;

      void epoch();
      real_value log_likelihood();
  };
}

#include <TrainingAlgorithm.tpp>
#endif