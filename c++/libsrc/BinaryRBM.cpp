#include <BinaryRBM.hpp>
using namespace rbm;

#include <cstddef>
#include <vector>
#include <random>

template<typename real_value>
const real_value BinaryRBM<real_value>::default_mu = real_value(0.);

template<typename real_value>
const real_value BinaryRBM<real_value>::default_sigma = real_value(1.);

template<typename real_value>
BinaryRBM<real_value>::BinaryRBM(std::size_t m, std::size_t n, std::mt19937& rng) :
  _m(m),
  _n(n),
  _b(m),
  _c(n),
  _w(n*m),
  _rng(rng)
{
  init_gaussian_b();
  init_constant_c();
  init_gaussian_w();
}

template<typename real_value>
BinaryRBM<real_value>::BinaryRBM(const std::vector<real_value>& b, const std::vector<real_value>& c, const std::vector<std::vector<real_value>>& w, std::mt19937& rng) :
  _m(b.size()),
  _n(c.size()),
  _b(b.begin(), b.end()),
  _c(c.begin(), c.end()),
  _rng(rng)
{
  _w.reserve(_n*_m);
  for (auto& w_i: w) {
    for (auto& w_ij: w_i) {
      _w.push_back(w_ij);
    }
  }
}

template<typename real_value>
BinaryRBM<real_value>::BinaryRBM(const std::vector<real_value>& b, const std::vector<real_value>& c, const std::vector<real_value>& w, std::mt19937& rng)  :
  _m(b.size()),
  _n(c.size()),
  _b(b.begin(), b.end()),
  _c(c.begin(), c.end()),
  _w(w.begin(), w.end()),
  _rng(rng)
{}

template<typename real_value>
void BinaryRBM<real_value>::init_gaussian_b(real_value mu, real_value sigma) {
  _b.resize(_m);
  std::normal_distribution<real_value> distribution(mu, sigma);
  for (auto& b_i: _b) {
    b_i = distribution(_rng); 
  }
}

template<typename real_value>
void BinaryRBM<real_value>::init_gaussian_c(real_value mu, real_value sigma) {
  _c.resize(_n);
  std::normal_distribution<real_value> distribution(mu, sigma);
  for (auto& c_j: _c) {
    c_j = distribution(_rng); 
  }
}

template<typename real_value>
void BinaryRBM<real_value>::init_fixed_b(const std::vector<real_value>& b) {
  _b = std::vector<real_value>(b.begin(), b.end());
}

template<typename real_value>
void BinaryRBM<real_value>::init_constant_c(real_value c) {
  _c.assign(_n, c);
}

template<typename real_value>
void BinaryRBM<real_value>::init_gaussian_w(real_value mu, real_value sigma) {
  _w.resize(_n*_m);
  std::normal_distribution<real_value> distribution(mu, sigma);
  for (auto& w_ij: _w) {
    w_ij = distribution(_rng); 
  }
}

// Explicit Instantiation
namespace rbm {
  template class BinaryRBM<float>;
  template class BinaryRBM<double>;
  template class BinaryRBM<long double>;
}