#ifndef __BinaryRBM_hpp__
#define __BinaryRBM_hpp__

#include <cstddef>
#include <vector>
#include <random>
#include <string>

/* The class is avaible with real_value in {float, double, long double} */

namespace rbm {
  
  template <typename real_value>
  class BinaryRBM {
    private:
      std::size_t _m, _n;
      std::vector<real_value> _b, _c, _w;

      static const real_value default_mu;
      static const real_value default_sigma;
    public:
      std::mt19937& _rng;
      
      // Constructors
      BinaryRBM(std::size_t m, std::size_t n, std::mt19937& rng);
      BinaryRBM(const std::vector<real_value>& b, const std::vector<real_value>& c, const std::vector<real_value>& w, std::mt19937& rng);
      BinaryRBM(const std::vector<real_value>& b, const std::vector<real_value>& c, const std::vector<std::vector<real_value>>& w, std::mt19937& rng);

      //Initializers: default values come from Hinton
      void init_gaussian_b(real_value mu = default_mu, real_value sigma = default_sigma);
      void init_fixed_b(const std::vector<real_value>& b);
      void init_gaussian_c(real_value mu = default_mu, real_value sigma = default_sigma);
      void init_constant_c(real_value c = 0.);
      void init_gaussian_w(real_value mu = default_mu, real_value sigma = default_sigma);

      // Files
      void load_from_file(std::string filename);
      void save_on_file(std::string filename) const;


      // Getters
      const std::size_t& n() const;
      const std::size_t& m() const;
      const std::vector<real_value>& b() const;
      const std::vector<real_value>& c() const;
      const std::vector<real_value>& w() const;
      const real_value& b(std::size_t i) const;
      const real_value& c(std::size_t j) const;
      const real_value& w(std::size_t i, std::size_t j) const;

      // Updaters
      void update_b(std::size_t i, real_value upd);
      void update_c(std::size_t j, real_value upd);
      void update_w(std::size_t i, std::size_t j, real_value upd);

      // Probabilities
      template<class Iterator>
      real_value prob_v(std::size_t i, Iterator h_begin) const;
      template<class Iterator>
      real_value prob_h(std::size_t j, Iterator v_begin) const;
      template<class Iterator>
      std::vector<real_value> vec_prob_v(Iterator h_begin) const;
      template<class Iterator>
      std::vector<real_value> vec_prob_h(Iterator v_begin) const;

      //Others
      template<class Iterator>
      real_value free_energy_v(Iterator v_begin) const;
  };

}

#include <BinaryRBM.tpp>
#endif