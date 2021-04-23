#ifndef __MarkovChain_hpp__
#define __MarkovChain_hpp__

#include <BinaryRBM.hpp>

#include <cstddef>
#include <vector>
#include <random>
#include <utility.cpp>

namespace rbm {
  template <typename real_value>
  class MarkovChain {
    private:
      BinaryRBM<real_value>& _rbm;
      std::vector<binary_value> _v, _h;
      std::mt19937& _rng;
      static const real_value default_init_probabality;
      std::minstd_rand fast_rng;
    public:
      // Constructors
      MarkovChain(BinaryRBM<real_value>& rbm, std::mt19937& rng);
      template<class Iterator>
      MarkovChain(BinaryRBM<real_value>& rbm, Iterator begin, Iterator end, std::mt19937& rng);

      //Initializers
      // TODO: move hardcoded value to approriete constant variables
      void init_random_h(real_value prob_one = default_init_probabality);
      void init_random_v(real_value prob_one = default_init_probabality);

      // Getters
      const std::vector<binary_value>& v() const;
      const std::vector<binary_value>& h() const;
      // I don't write getters for iterators because they can be easilly got from these.

      //Setters
      template<class Iterator>
      void set_v(Iterator begin, Iterator end);
      template<class Iterator>
      void set_h(Iterator begin, Iterator end);
      
      // Next Step
      void next_step_v();
      void next_step_h();

      // Next Steps: shortcuts
      void evolve(unsigned times);
  };
}

template<typename real_value>
const real_value MarkovChain<real_value>::default_init_probabality = real_value(0.5);

#include <MarkovChain.tpp>
#endif