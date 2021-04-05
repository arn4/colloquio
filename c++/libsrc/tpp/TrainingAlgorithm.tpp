#include <TrainingAlgorithm.hpp>
#include <utility.cpp>

#include <cstddef>

template<typename real_value, std::size_t features_size, std::size_t batch_size>
TrainingAlgorithm<real_value, features_size, batch_size>::TrainingAlgorithm (
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  real_value lr,
  real_value wd,
  real_value m
) :
  _rbm(rbm),
  _training_set(training_set),
  learning_rate(lr),
  weight_decay(wd),
  momentum(m)
{
  _last_update_b.assign(_rbm.m(), real_value(0.));
  _last_update_b.assign(_rbm.n(), real_value(0.));
  _last_update_w.assign(_rbm.m() * _rbm.n(), real_value(0.));
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void TrainingAlgorithm<real_value, features_size, batch_size>::train_on_batch(std::size_t b) {
  batch_precomputing(b);

  // Applying momentum and weight decay
  for (std::size_t i = 0; i < _rbm.m(); i++) {
    _last_update_b[i] *= momentum;
    _last_update_b[i] -= weight_decay * _rbm.b(i);
  }
  for (std::size_t j = 0; j < _rbm.n(); j++) {
    _last_update_c[j] *= momentum;
    _last_update_c[j] -= weight_decay * _rbm.b(j);
  }
  for (std::size_t i = 0; i < _rbm.m(); i++) {
    for (std::size_t j = 0; j < _rbm.n(); j++) {
      _last_update_w[features_size*i + j] *= momentum;
      _last_update_w[features_size*i + j] -= weight_decay * _rbm.w(i,j);
    }
  }

  const TrainingBatch<features_size, batch_size>& batch = _training_set.batch(b);
  for (std::size_t k = 0; k < batch_size; k++) {
    // Compute log-likelihood of b_i
    for (std::size_t i = 0; i < _rbm.m(); i++) {
      _last_update_b[i] = bool2real<real_value>(batch.get_element(k,i)) + b_second_term(i, k);
    }

    // Compute log-likelihood of c_j and w_ij; all together to optimize 
    for (std::size_t j = 0; j < _rbm.n(); j++) {
      real_value probability_hj = _rbm.prob_h(j, batch.get_iterator(k));
      _last_update_c[j] = probability_hj + c_second_term(j, k);
      for (std::size_t i = 0; i < _rbm.m(); i++) {
        _last_update_w[features_size*i + j] = bool2real<real_value>(batch.get_element(k,i)) * probability_hj + w_second_term(i, j, k);
      }
    }
  }

  for (std::size_t i = 0; i < _rbm.m(); i++) {
    _rbm.update_b(i, _last_update_b[i]);
  }
  for (std::size_t j = 0; j < _rbm.n(); j++) {
    _rbm.update_c(j, _last_update_c[j]);
  }
  for (std::size_t i = 0; i < _rbm.m(); i++) {
    for (std::size_t j = 0; j < _rbm.n(); j++) {
      _rbm.update_w(i, j, _last_update_w[i*features_size+j]);
    }
  }
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void TrainingAlgorithm<real_value, features_size, batch_size>::epoch() {
  epoch_precomputing();

  for (std::size_t b = 0; b < batch_size; b++) {
    train_on_batch(b);
  }
}