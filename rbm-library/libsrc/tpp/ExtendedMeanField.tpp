#include <ExtendedMeanField.hpp>
#include <TrainingAlgorithm.hpp>
#include <MarcovChain.hpp>
#include <utility.cpp>

#include <iostream>
#include <cstddef>
#include <algorithm>
#include <cassert>
#include <cmath>







// Implementations of calculations:
template<typename real_value>
inline void first_order_convergence_step(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm) {
  std::vector<real_value> arg_mv(rbm.m(), real_value(0.));
  std::vector<real_value> arg_mh(rbm.n(), real_value(0.));
  
  for (unsigned j = 0; j < rbm.n(); j++) {
    arg_mh[j] += rbm.c(j);
    for (unsigned i = 0; i < rbm.m(); i++) {
      arg_mh[j] += rbm.w(i,j)*mv[i];
    }
    mh[j] = sigmoid(arg_mh[j]);
  }

  for (unsigned i = 0; i < rbm.m(); i++) {
    arg_mv[i] += rbm.b(i);
    for (unsigned j = 0; j < rbm.n(); j++) {
      arg_mv[i] += rbm.w(i,j)*mh[j];
    }
    mv[i] = sigmoid(arg_mv[i]);
  }
}

template<typename real_value>
inline real_value compute_first_order_weight_update(std::size_t i, std::size_t j, std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>&) {
  return -mv[i]*mh[j];
}

template<typename real_value>
inline real_value first_order_gibbs_energy(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm) {
  real_value gf = 0.;
  for (std::size_t i = 0; i < rbm.m(); i++) {
    gf += safe_xlogx(mv[i]) + safe_xlogx(real_value(1.)-mv[i]) - rbm.b(i) * mv[i];
  }
  for (std::size_t j = 0; j < rbm.n(); j++) {
    gf += safe_xlogx(mh[j]) + safe_xlogx(real_value(1.)-mh[j] ) - rbm.c(j) * mh[j];
  }
  for (std::size_t i = 0; i < rbm.m(); i++) {
    for (std::size_t j = 0; j < rbm.n(); j++) {
      gf -= mv[i] * rbm.w(i,j) * mh[j];
    }
  }
  return gf;
}

template<typename real_value>
inline void second_order_convergence_step(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm) {
  std::vector<real_value> arg_mh(rbm.n(), real_value(0.));
  std::vector<real_value> arg_mv(rbm.m(), real_value(0.));
  
  for (unsigned j = 0; j < rbm.n(); j++) {
    arg_mh[j] += rbm.c(j);
    for (unsigned i = 0; i < rbm.m(); i++) {
      arg_mh[j] +=
      rbm.w(i,j) * mv[i] * (
        real_value(1.) +
        rbm.w(i,j) * (
          real_value(1.) -
          mv[i]
        ) * (
          real_value(.5)-
          mh[j]
        )
      );
    }
    mh[j] = sigmoid(arg_mh[j]);
  }

  for (unsigned i = 0; i < rbm.m(); i++) {
    arg_mv[i] += rbm.b(i);
    for (unsigned j = 0; j < rbm.n(); j++) {
      arg_mv[i] +=
      rbm.w(i,j) * mh[j] * (
        real_value(1.) +
        rbm.w(i,j) * (
          real_value(1.)-
          mh[j]
        ) * (
          real_value(.5)-
          mv[i]
        )
      );
    }
    mv[i] = sigmoid(arg_mv[i]);
  }
}

template<typename real_value>
inline real_value compute_second_order_weight_update(std::size_t i, std::size_t j, std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm) {
  return 
  -mv[i]*mh[j] * (
    real_value(1.) +
    rbm.w(i,j) * (
      real_value(1.)-
      mv[i]
    ) * (
      real_value(1.)-
      mh[j]
    )
  );
}

template<typename real_value>
inline real_value second_order_gibbs_energy(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm) {
  real_value gf = first_order_gibbs_energy(mv, mh, rbm);
  for (std::size_t i = 0; i < rbm.m(); i++) {
    for (std::size_t j = 0; j < rbm.n(); j++) {
      gf += -mv[i]*mh[j] * rbm.w(i,j) * rbm.w(i,j) * (real_value(1.)-mv[i]) * (real_value(1.)-mh[j])/ real_value(2.);
    }
  }
  return gf;
}

template<typename real_value>
inline void third_order_convergence_step(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm) {
  std::vector<real_value> arg_mv(rbm.m(), real_value(0.));
  std::vector<real_value> arg_mh(rbm.n(), real_value(0.));
  
  for (unsigned j = 0; j < rbm.n(); j++) {
    arg_mh[j] += rbm.c(j);
    for (unsigned i = 0; i < rbm.m(); i++) {
      arg_mh[j] +=
      rbm.w(i,j) * mv[i] * (
        real_value(1.) +
        rbm.w(i,j) * (
          real_value(1.) -
          mv[i]
        ) * (
          (
            real_value(.5)-
            mh[j]
          ) +
          rbm.w(i,j) * (
            2*mh[j] * (
              mh[j] -
              real_value(1.)
            ) +
            real_value(1.)/real_value(3.)
          ) * (
            real_value(.5)-
            mv[i]
          )
        )
      );
    }
    mh[j] = sigmoid(arg_mh[j]);
  }

  for (unsigned i = 0; i < rbm.m(); i++) {
    arg_mv[i] += rbm.b(i);
    for (unsigned j = 0; j < rbm.n(); j++) {
      arg_mv[i] +=
      rbm.w(i,j) * mh[j] * (
        real_value(1.) +
        rbm.w(i,j) * (
          real_value(1.)-
          mh[j]
        ) * (
          (
            real_value(.5)-
            mv[i]
          ) +
          rbm.w(i,j) * (
            2*mv[i] * (
              mv[i] -
              real_value(1.)
            ) +
            real_value(1.)/real_value(3.)
          ) * (
            real_value(.5)-
            mh[j]
          )
        )
      );
    }
    mv[i] = sigmoid(arg_mv[i]);
  }
}

template<typename real_value>
inline real_value compute_third_order_weight_update(std::size_t i, std::size_t j, std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm) {
  return 
  -mv[i]*mh[j] * (
    real_value(1.) +
    rbm.w(i,j) * (
      real_value(1.)-
      mv[i]
    ) * (
      real_value(1.)-
      mh[j]
    ) * (
      real_value(1.) +
      2 * rbm.w(i,j) * (
        real_value(.5)-
        mv[i]
      ) * (
        real_value(.5)-
        mh[j]
      )
    )
  );
}

template<typename real_value>
inline real_value third_order_gibbs_energy(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm) {
  real_value gf = first_order_gibbs_energy(mv, mh, rbm);
  for (std::size_t i = 0; i < rbm.m(); i++) {
    for (std::size_t j = 0; j < rbm.n(); j++) {
      gf += -mv[i]*mh[j] * rbm.w(i,j) * rbm.w(i,j) * (real_value(1.)-mv[i]) * (real_value(1.)-mh[j]) * 
            (real_value(1)/real_value(2) + 
            real_value(2)/real_value(3) * rbm.w(i,j) * (real_value(1)/real_value(2)-mv[i]) * (real_value(1)/real_value(2)-mh[j]));
    }
  }
  return gf;
}

// Random magnetization
template<typename real_value>
inline void random_m(std::vector<real_value>& m, std::mt19937& rng) {
  std::uniform_real_distribution<real_value> distribution(real_value(0.),real_value(1.));
  for (auto& mx: m) {
    mx = distribution(rng);
  }
}











template<typename real_value, std::size_t features_size, std::size_t batch_size>
MeanField<real_value, features_size, batch_size>::MeanField(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  std::mt19937& rng,
  unsigned l,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  TrainingAlgorithm<real_value, features_size, batch_size>(rbm, training_set, lr, wd, m),
  _rng(rng),
  _l(l),
  _k(k),
  b_update_table(rbm.m()),
  c_update_table(rbm.n()),
  w_update_table(rbm.n()*rbm.m()),
  _mv(l, std::vector<real_value>(rbm.m())),
  _mh(l, std::vector<real_value>(rbm.n()))
{
  for (auto& mv: _mv){
    random_m(mv, _rng);
  }

  for (auto& mh: _mh){
    random_m(mh, _rng);
  }
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void MeanField<real_value, features_size, batch_size>::init_m() {
  for (auto& mv: _mv) {
    random_m(mv, _rng);
  }
  for (auto& mh: _mh) {
    random_m(mh, _rng);
  }
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
void MeanField<real_value, features_size, batch_size>::batch_precomputing(std::size_t) {
  init_m();
  std::vector<real_value> energies_weights(_l);
  real_value total_weight = 0.;
  for (std::size_t g = 0; g < _l; g++) {
    // Converge magnetization
    for (unsigned t = 1; t <= _k; t++) {
      convergence_step(g);
    }
    energies_weights[g] = std::exp(-gibbs_energy(g) / (MeanField::_rbm.m()+MeanField::_rbm.n()));
    total_weight += energies_weights[g];
  }

  for (auto& e: energies_weights) {
    e /= total_weight;
  }

  for (unsigned i = 0; i < MeanField::_rbm.m(); i++) {
    b_update_table[i] = 0.;
    for (std::size_t g = 0; g < _l; g++) {
      b_update_table[i] += energies_weights[g] * (-_mv[g][i]);
    }
  }
  for (unsigned j = 0; j < MeanField::_rbm.n(); j++) {
    c_update_table[j] = 0.;
    for (std::size_t g = 0; g < _l; g++) {
      c_update_table[j] += energies_weights[g] * (-_mh[g][j]);
    }
  }
  for (unsigned i = 0; i < MeanField::_rbm.m(); i++) {
    for (unsigned j = 0; j < MeanField::_rbm.n(); j++) {
      w_update_table[MeanField::_rbm.n()*i+j] = 0.;
      for (std::size_t g = 0; g < _l; g++) {
        w_update_table[MeanField::_rbm.n()*i+j] += energies_weights[g] * compute_weight_update(i, j, g);
      }
    }
  }

}


template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void MeanField<real_value, features_size, batch_size>::epoch_precomputing(std::size_t) {}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value MeanField<real_value, features_size, batch_size>::w_second_term(std::size_t i, std::size_t j, std::size_t) {
  return w_update_table[MeanField::_rbm.n()*i+j];
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value MeanField<real_value, features_size, batch_size>::b_second_term(std::size_t i, std::size_t) {
  return b_update_table[i];
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value MeanField<real_value, features_size, batch_size>::c_second_term(std::size_t j, std::size_t) {
  return c_update_table[j];
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value MeanField<real_value, features_size, batch_size>::compute_weight_update(std::size_t i, std::size_t j, std::size_t r) {
  return compute_first_order_weight_update(i, j, _mv[r], _mh[r], MeanField::_rbm);
}

template<typename real_value, std::size_t features_size, std::size_t batch_size> 
inline real_value MeanField<real_value, features_size, batch_size>::gibbs_energy(std::size_t r) {
  return first_order_gibbs_energy(_mv[r], _mh[r], MeanField::_rbm);
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void MeanField<real_value, features_size, batch_size>::convergence_step(std::size_t r) {
  first_order_convergence_step(_mv[r], _mh[r], MeanField::_rbm);
}

// Persistent MeanField
template<typename real_value, std::size_t features_size, std::size_t batch_size>
PersistentMeanField<real_value, features_size, batch_size>::PersistentMeanField(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  std::mt19937& rng,
  unsigned l,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  MeanField<real_value, features_size, batch_size>(rbm, training_set, rng, l, k, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void PersistentMeanField<real_value, features_size, batch_size>::init_m() {}


// ThoulessAndersonPalmer at Order2
template<typename real_value, std::size_t features_size, std::size_t batch_size>
TAP2<real_value, features_size, batch_size>::TAP2(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  std::mt19937& rng,
  unsigned l,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  MeanField<real_value, features_size, batch_size>(rbm, training_set, rng, l, k, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value TAP2<real_value, features_size, batch_size>::compute_weight_update(std::size_t i, std::size_t j, std::size_t r) {
  return compute_second_order_weight_update(i, j, TAP2::_mv[r], TAP2::_mh[r], TAP2::_rbm);
}

template<typename real_value, std::size_t features_size, std::size_t batch_size> 
inline real_value TAP2<real_value, features_size, batch_size>::gibbs_energy(std::size_t r) {
  return second_order_gibbs_energy(TAP2::_mv[r], TAP2::_mh[r], TAP2::_rbm);
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void TAP2<real_value, features_size, batch_size>::convergence_step(std::size_t r) {
  second_order_convergence_step(TAP2::_mv[r], TAP2::_mh[r], TAP2::_rbm);
}


// Persistent ThoulessAndersonPalmer at Order2
template<typename real_value, std::size_t features_size, std::size_t batch_size>
PersistentTAP2<real_value, features_size, batch_size>::PersistentTAP2(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  std::mt19937& rng,
  unsigned l,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  TAP2<real_value, features_size, batch_size>(rbm, training_set, rng, l, k, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void PersistentTAP2<real_value, features_size, batch_size>::init_m() {}

// ThoulessAndersonPalmer at Order3
template<typename real_value, std::size_t features_size, std::size_t batch_size>
TAP3<real_value, features_size, batch_size>::TAP3(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  std::mt19937& rng,
  unsigned l,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  MeanField<real_value, features_size, batch_size>(rbm, training_set, rng, l, k, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline real_value TAP3<real_value, features_size, batch_size>::compute_weight_update(std::size_t i, std::size_t j, std::size_t r) {
  return compute_third_order_weight_update(i, j, TAP3::_mv[r], TAP3::_mh[r], TAP3::_rbm);
}

template<typename real_value, std::size_t features_size, std::size_t batch_size> 
inline real_value TAP3<real_value, features_size, batch_size>::gibbs_energy(std::size_t r) {
  return third_order_gibbs_energy(TAP3::_mv[r], TAP3::_mh[r], TAP3::_rbm);
}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void TAP3<real_value, features_size, batch_size>::convergence_step(std::size_t r) {
  third_order_convergence_step(TAP3::_mv[r], TAP3::_mh[r], TAP3::_rbm);
}

// Persistent ThoulessAndersonPalmer at Order3
template<typename real_value, std::size_t features_size, std::size_t batch_size>
PersistentTAP3<real_value, features_size, batch_size>::PersistentTAP3(
  BinaryRBM<real_value>& rbm,
  TrainingSet<features_size, batch_size>& training_set,
  std::mt19937& rng,
  unsigned l,
  unsigned k,
  real_value lr,
  real_value wd,
  real_value m
) :
  TAP3<real_value, features_size, batch_size>(rbm, training_set, rng, l, k, lr, wd, m)
{}

template<typename real_value, std::size_t features_size, std::size_t batch_size>
inline void PersistentTAP3<real_value, features_size, batch_size>::init_m() {}



