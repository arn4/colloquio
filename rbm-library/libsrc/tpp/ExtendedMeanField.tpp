#include <ExtendedMeanField.hpp>
#include <TrainingAlgorithm.hpp>
#include <MarcovChain.hpp>
#include <utility.cpp>

#include <iostream>
#include <cstddef>
#include <algorithm>
#include <cassert>


template<typename real_value, std::size_t features_size, std::size_t batch_size>
MeanField<real_value, features_size, batch_size>::MeanField(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  TrainingAlgorithm<real_value, features_size, batch_size>(rbm, training_set, lr, wd, m),
  //_rng(rng),
  mv(rbm.m(), real_value(0.5)),
  mh(rbm.n(), real_value(0.5)),
  weight_update_table(rbm.n()*rbm.m()),
  _k(k)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void MeanField<real_value, features_size, batch_size>::init_m() {
  fill(mv.begin(), mv.end(), real_value(0.5));
  fill(mh.begin(), mh.end(), real_value(0.5));
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void MeanField<real_value, features_size, batch_size>::batch_precomputing(std::size_t) {
  init_m();
  for (unsigned t = 1; t <= _k; t++) {
    compute_logistic_arguments();
    for (auto& mi: mv) {
      mi = sigmoid(mi);
    }
    for (auto& mj: mh) {
      mj = sigmoid(mj);
    }
  }
  for (unsigned i = 0; i < MeanField::_rbm.m(); i++) {
    for (unsigned j = 0; j < MeanField::_rbm.n(); j++) {
      weight_update_table[MeanField::_rbm.n()*i+j] = compute_weight_update(i,j);
    }
  }
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void MeanField<real_value, features_size, batch_size>::epoch_precomputing(std::size_t) {}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value MeanField<real_value, features_size, batch_size>::w_second_term(std::size_t i, std::size_t j, std::size_t) {
  return weight_update_table[MeanField::_rbm.n()*i+j];
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value MeanField<real_value, features_size, batch_size>::b_second_term(std::size_t i, std::size_t) {
  return -mv[i];
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value MeanField<real_value, features_size, batch_size>::c_second_term(std::size_t j, std::size_t) {
  return -mh[j];
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void MeanField<real_value, features_size, batch_size>::compute_logistic_arguments() {
  std::vector<real_value> arg_mv(MeanField::_rbm.m(), real_value(0.));
  for (unsigned i = 0; i < MeanField::_rbm.m(); i++) {
    arg_mv[i] += MeanField::_rbm.b(i);
    for (unsigned j = 0; j < MeanField::_rbm.n(); j++) {
      arg_mv[i] += MeanField::_rbm.w(i,j)*mh[j];
    }
  }
  
  std::vector<real_value> arg_mh(MeanField::_rbm.n(), real_value(0.));
  for (unsigned j = 0; j < MeanField::_rbm.n(); j++) {
    arg_mh[j] += MeanField::_rbm.c(j);
    for (unsigned i = 0; i < MeanField::_rbm.m(); i++) {
      arg_mh[j] += MeanField::_rbm.w(i,j)*mv[i];
    }
  }
  mv = arg_mv;
  mh = arg_mh;
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value MeanField<real_value, features_size, batch_size>::compute_weight_update(std::size_t i, std::size_t j) {
  return -mv[i]*mh[j];
}

// Persistent MeanField
template<typename real_value, std::size_t features_size, std::size_t batch_size>
PersistentMeanField<real_value, features_size, batch_size>::PersistentMeanField(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  MeanField<real_value, features_size, batch_size>(rbm, training_set, k, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void PersistentMeanField<real_value, features_size, batch_size>::init_m() {}



// ThoulessAndersonPalmer at Order3
template<typename real_value, std::size_t features_size, std::size_t batch_size>
TAP2<real_value, features_size, batch_size>::TAP2(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  MeanField<real_value, features_size, batch_size>(rbm, training_set, k, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void TAP2<real_value, features_size, batch_size>::compute_logistic_arguments() {
  std::vector<real_value> arg_mv(TAP2::_rbm.m(), real_value(0.));
  for (unsigned i = 0; i < TAP2::_rbm.m(); i++) {
    arg_mv[i] += TAP2::_rbm.b(i);
    for (unsigned j = 0; j < TAP2::_rbm.n(); j++) {
      arg_mv[i] +=
      TAP2::_rbm.w(i,j) * TAP2::mh[j] * (
        real_value(1.) +
        TAP2::_rbm.w(i,j) * (
          real_value(1.)-
          TAP2::mh[j]
        ) * (
          real_value(.5)-
          TAP2::mv[i]
        )
      );
    }
  }
  
  std::vector<real_value> arg_mh(TAP2::_rbm.n(), real_value(0.));
  for (unsigned j = 0; j < TAP2::_rbm.n(); j++) {
    arg_mh[j] += TAP2::_rbm.c(j);
    for (unsigned i = 0; i < TAP2::_rbm.m(); i++) {
      arg_mh[j] +=
      TAP2::_rbm.w(i,j) * TAP2::mv[i] * (
        real_value(1.) +
        TAP2::_rbm.w(i,j) * (
          real_value(1.) -
          TAP2::mv[i]
        ) * (
          real_value(.5)-
          TAP2::mh[j]
        )
      );
    }
  }
  TAP2::mv = arg_mv;
  TAP2::mh = arg_mh;
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value TAP2<real_value, features_size, batch_size>::compute_weight_update(std::size_t i, std::size_t j) {
  return 
  -TAP2::mv[i]*TAP2::mh[j] * (
    real_value(1.) +
    TAP2::_rbm.w(i,j) * (
      real_value(1.)-
      TAP2::mv[i]
    ) * (
      real_value(1.)-
      TAP2::mh[j]
    )
  );
}

// Persistent ThoulessAndersonPalmer at Order2
template<typename real_value, std::size_t features_size, std::size_t batch_size>
PersistentTAP2<real_value, features_size, batch_size>::PersistentTAP2(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  TAP2<real_value, features_size, batch_size>(rbm, training_set, k, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void PersistentTAP2<real_value, features_size, batch_size>::init_m() {}




// ThoulessAndersonPalmer at Order3
template<typename real_value, std::size_t features_size, std::size_t batch_size>
TAP3<real_value, features_size, batch_size>::TAP3(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  MeanField<real_value, features_size, batch_size>(rbm, training_set, k, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void TAP3<real_value, features_size, batch_size>::compute_logistic_arguments() {
  std::vector<real_value> arg_mv(TAP3::_rbm.m(), real_value(0.));
  for (unsigned i = 0; i < TAP3::_rbm.m(); i++) {
    arg_mv[i] += TAP3::_rbm.b(i);
    for (unsigned j = 0; j < TAP3::_rbm.n(); j++) {
      arg_mv[i] +=
      TAP3::_rbm.w(i,j) * TAP3::mh[j] * (
        real_value(1.) +
        TAP3::_rbm.w(i,j) * (
          real_value(1.)-
          TAP3::mh[j]
        ) * (
          (
            real_value(.5)-
            TAP3::mv[i]
          ) +
          TAP3::_rbm.w(i,j) * (
            2*TAP3::mv[i] * (
              TAP3::mv[i] -
              real_value(1.)
            ) +
            real_value(1.)/real_value(3.)
          ) * (
            real_value(.5)-
            TAP3::mh[j]
          )
        )
      );
    }
  }
  
  std::vector<real_value> arg_mh(TAP3::_rbm.n(), real_value(0.));
  for (unsigned j = 0; j < TAP3::_rbm.n(); j++) {
    arg_mh[j] += TAP3::_rbm.c(j);
    for (unsigned i = 0; i < TAP3::_rbm.m(); i++) {
      arg_mh[j] +=
      TAP3::_rbm.w(i,j) * TAP3::mv[i] * (
        real_value(1.) +
        TAP3::_rbm.w(i,j) * (
          real_value(1.) -
          TAP3::mv[i]
        ) * (
          (
            real_value(.5)-
            TAP3::mh[j]
          ) +
          TAP3::_rbm.w(i,j) * (
            2*TAP3::mh[j] * (
              TAP3::mh[j] -
              real_value(1.)
            ) +
            real_value(1.)/real_value(3.)
          ) * (
            real_value(.5)-
            TAP3::mv[i]
          )
        )
      );
    }
  }
  TAP3::mv = arg_mv;
  TAP3::mh = arg_mh;
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value TAP3<real_value, features_size, batch_size>::compute_weight_update(std::size_t i, std::size_t j) {
  return 
  -TAP3::mv[i]*TAP3::mh[j] * (
    real_value(1.) +
    TAP3::_rbm.w(i,j) * (
      real_value(1.)-
      TAP3::mv[i]
    ) * (
      real_value(1.)-
      TAP3::mh[j]
    ) * (
      real_value(1.) +
      2 * TAP3::_rbm.w(i,j) * (
        real_value(.5)-
        TAP3::mv[i]
      ) * (
        real_value(.5)-
        TAP3::mh[j]
      )
    )
  );
}

// Persistent ThoulessAndersonPalmer at Order3
template<typename real_value, std::size_t features_size, std::size_t batch_size>
PersistentTAP3<real_value, features_size, batch_size>::PersistentTAP3(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  TAP3<real_value, features_size, batch_size>(rbm, training_set, k, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void PersistentTAP3<real_value, features_size, batch_size>::init_m() {}

