#include <TrainingSet.hpp>
using namespace rbm;

#include <cstddef>
#include <vector>
#include <cassert>
#include <array>
#include <algorithm>

template<std::size_t features_size, std::size_t batch_size>
TrainingBatch<features_size, batch_size>::TrainingBatch(const std::vector<std::vector<bool>>& batch) {
  assert(batch.size() == batch_size);
  std::size_t ki = 0;
  for (auto& d: batch) {
    assert(d.size() ==features_size);
    for(bool f: d) {
      data[ki] = binary_value(f);
      ++ki;
    }
  }
}

template<std::size_t features_size, std::size_t batch_size>
inline auto TrainingBatch<features_size, batch_size>::get_iterator(std::size_t k) const {
  return data.begin() + k*features_size;
}

template<std::size_t features_size, std::size_t batch_size>
inline binary_value TrainingBatch<features_size, batch_size>::get_element(std::size_t k, std::size_t i) const {
  return data[k*features_size+i];
}

template<std::size_t features_size, std::size_t batch_size>
inline const TrainingBatch<features_size, batch_size>& TrainingSet<features_size, batch_size>::batch(std::size_t s) const {
  return _training_set[s];
}

template<std::size_t features_size, std::size_t batch_size>
TrainingSet<features_size, batch_size>::TrainingSet(const std::vector<std::vector<bool>>& training_set) {
  assert(not training_set.empty());
  for (auto& ds: training_set) {
    assert(ds.size() == features_size);
  }

  _training_set.resize(training_set.size()/batch_size);
  for (std::size_t i = 0; i + batch_size <= training_set.size(); i += batch_size) {
    _training_set.at(i/batch_size) = TrainingBatch<features_size,batch_size>(
                                      std::vector<std::vector<bool>>(
                                        training_set.begin()+long(i),
                                        training_set.begin()+long(i)+batch_size
                                      )
                                    );
  }
}

template<std::size_t features_size, std::size_t batch_size>
TrainingSet<features_size, batch_size>::TrainingSet(const std::vector<std::vector<std::vector<bool>>>& classes, std::mt19937& rng) {
  // TODO: shuffle class. This is not working!
  // for(auto& cl: classes) {
  //   std::shuffle(cl.begin(), cl.end(), rng);
  // }

  std::vector<std::size_t> last_index(classes.size(), 0);
  while (true) {
    std::vector<std::vector<bool>> candidate_batch;
    for (std::size_t c = 0; c < classes.size(); c++) {
      if (classes[c].size() == last_index[c]) {
        return;
      }
      candidate_batch.push_back(classes[c][last_index[c]]);
      last_index[c]++;
    }
    std::shuffle(candidate_batch.begin(), candidate_batch.end(), rng);
    _training_set.push_back(TrainingBatch<features_size, batch_size>(candidate_batch));
  }
}

template<std::size_t features_size, std::size_t batch_size>
inline std::size_t TrainingSet<features_size, batch_size>::num_of_batches() {
  return _training_set.size();
}
