#ifndef __ExtendedMeanField_hpp__
#define __ExtendedMeanField_hpp__

#include <BinaryRBM.hpp>
#include <TrainingSet.hpp>
#include <TrainingAlgorithm.hpp>
#include <MarkovChain.hpp>

#include <cstddef>
#include <vector>

namespace rbm {
  // Calculations:
  // template<typename real_value>
  // void first_order_convergence_step(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm);

  // template<typename real_value>
  // real_value compute_first_order_weight_update(std::size_t i, std::size_t j, std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm);

  // template<typename real_value>
  // inline real_value first_order_gibbs_energy(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm);

  // template<typename real_value>
  // void second_order_convergence_step(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm);

  // template<typename real_value>
  // real_value compute_second_order_weight_update(std::size_t i, std::size_t j, std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm);

  // template<typename real_value>
  // real_value second_order_gibbs_energy(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm);

  // template<typename real_value>
  // void third_order_convergence_step(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm);
    
  // template<typename real_value>
  // real_value compute_third_order_weight_update(std::size_t i, std::size_t j, std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm);

  // template<typename real_value>
  // real_value third_order_gibbs_energy(std::vector<real_value>& mv, std::vector<real_value>& mh, BinaryRBM<real_value>& rbm);

  // template<typename real_value>
  // void random_m(std::vector<real_value>& m, std::mt19937& rng);

  template<
    typename real_value,
    std::size_t features_size,
    std::size_t batch_size
  >
  class MeanField: public TrainingAlgorithm<real_value, features_size, batch_size> {
    private:
      std::mt19937& _rng;
      unsigned _l;
      unsigned _k;
      std::vector<real_value> b_update_table;
      std::vector<real_value> c_update_table;
      std::vector<real_value> w_update_table;
    protected:
      std::vector<std::vector<real_value>> _mv;
      std::vector<std::vector<real_value>> _mh;
      virtual void init_m();
      virtual void convergence_step(std::size_t r);
      virtual real_value compute_weight_update(std::size_t i, std::size_t j, std::size_t r);
      virtual real_value gibbs_energy(std::size_t r);
    public:
      MeanField(
        BinaryRBM<real_value>& rbm,
        TrainingSet<features_size, batch_size>& training_set,
        std::mt19937& rng,
        unsigned l,
        unsigned k,
        real_value lr = 0.,
        real_value wd = 0.,
        real_value m = 0.
      );

      void batch_precomputing(std::size_t b);
      void epoch_precomputing(std::size_t e);

      real_value w_second_term(std::size_t i, std::size_t j, std::size_t k);
      real_value b_second_term(std::size_t i, std::size_t k);
      real_value c_second_term(std::size_t j, std::size_t k);
  };

  template<
    typename real_value,
    std::size_t features_size,
    std::size_t batch_size
  >
  class PersistentMeanField: public MeanField<real_value, features_size, batch_size> {
    protected:
      void init_m();
    public:
      PersistentMeanField(
        BinaryRBM<real_value>& rbm,
        TrainingSet<features_size, batch_size>& training_set,
        std::mt19937& rng,
        unsigned l,
        unsigned k,
        real_value lr = 0.,
        real_value wd = 0.,
        real_value m = 0.
      );
  };

  template<
    typename real_value,
    std::size_t features_size,
    std::size_t batch_size
  >
  class TAP2: public MeanField<real_value, features_size, batch_size> {
    protected:
      void convergence_step(std::size_t r);
      real_value compute_weight_update(std::size_t i, std::size_t j, std::size_t r);
      real_value gibbs_energy(std::size_t r);
    public:
      TAP2(
        BinaryRBM<real_value>& rbm,
        TrainingSet<features_size, batch_size>& training_set,
        std::mt19937& rng,
        unsigned l,
        unsigned k,
        real_value lr = 0.,
        real_value wd = 0.,
        real_value m = 0.
      );
  };

  template<
    typename real_value,
    std::size_t features_size,
    std::size_t batch_size
  >
  class PersistentTAP2:
    // I know that the correct way would be using virtual inheritance to break the diamond,
    // but at the moment I prefer avoid that overcomplicated construct (moreover it is slower use virtual)
    // StackOverflow ref: https://stackoverflow.com/questions/67076911/multiple-inheritance-in-c-choose-from-which-class-take-the-member/67077036#67077036

    // public PersistentMeanField<real_value, features_size, batch_size>,
    public TAP2<real_value, features_size, batch_size>
  {
    protected:
      void init_m();
    public:
      PersistentTAP2(
        BinaryRBM<real_value>& rbm,
        TrainingSet<features_size, batch_size>& training_set,
        std::mt19937& rng,
        unsigned l,
        unsigned k,
        real_value lr = 0.,
        real_value wd = 0.,
        real_value m = 0.
      );
  };

  template<
    typename real_value,
    std::size_t features_size,
    std::size_t batch_size
  >
  class TAP3: public MeanField<real_value, features_size, batch_size> {
    protected:
      void convergence_step(std::size_t r);
      real_value compute_weight_update(std::size_t i, std::size_t j, std::size_t r);
      real_value gibbs_energy(std::size_t r);
    public:
      TAP3(
        BinaryRBM<real_value>& rbm,
        TrainingSet<features_size, batch_size>& training_set,
        std::mt19937& rng,
        unsigned l,
        unsigned k,
        real_value lr = 0.,
        real_value wd = 0.,
        real_value m = 0.
      );
  };

  template<
    typename real_value,
    std::size_t features_size,
    std::size_t batch_size
  >
  class PersistentTAP3:
    public TAP3<real_value, features_size, batch_size>
  {
    protected:
      void init_m();
    public:
      PersistentTAP3(
        BinaryRBM<real_value>& rbm,
        TrainingSet<features_size, batch_size>& training_set,
        std::mt19937& rng,
        unsigned l,
        unsigned k,
        real_value lr = 0.,
        real_value wd = 0.,
        real_value m = 0.
      );
  };
}

#include <ExtendedMeanField.tpp>
#endif