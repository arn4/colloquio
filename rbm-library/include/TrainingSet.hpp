#ifndef __TrainingSet_hpp__
#define __TrainingSet_hpp__

#include <cstddef>
#include <vector>
#include <random>
#include <array>
#include <utility.cpp>


namespace rbm {
  
  template<std::size_t features_size, std::size_t batch_size>
  class TrainingBatch {
    private:
      std::array<binary_value, features_size*batch_size> data;
    public:
      TrainingBatch() {};
      TrainingBatch(const std::vector<std::vector<bool>>& batch);

      auto get_iterator(std::size_t k) const;

      binary_value get_element(std::size_t k, std::size_t i) const;

  };

  template<std::size_t features_size, std::size_t batch_size>
  class TrainingSet {
    private:
      std::vector<TrainingBatch<features_size, batch_size>> _training_set;
    public:
      TrainingSet(const std::vector<std::vector<bool>>& training_set);
      TrainingSet(const std::vector<std::vector<std::vector<bool>>>& classes, std::mt19937& rng);

      const TrainingBatch<features_size, batch_size>& batch(std::size_t s) const;
      std::size_t num_of_batches();
  };

}

#include <TrainingSet.tpp>
#endif