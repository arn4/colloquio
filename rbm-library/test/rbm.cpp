#include <iostream>
#include <random>

#include <BernoulliRBM.hpp>


using namespace std;
using namespace rbm;

const unsigned SEED = 64770;
using real_value = double;

int main() {
  mt19937 rng(SEED);
  vector<real_value> b = {1., 3.};
  vector<real_value> c = {-2., -1.};
  vector<vector<real_value>>  w = {{2.,-6.}, {8.,-2.}};

  BernoulliRBM<real_value> rbm(b, c, w, rng);

  vector<bool> ii = {true, true};
  vector<bool> io = {true, false};
  vector<bool> oi = {false, true};
  vector<bool> oo = {false, false};

  clog << rbm.prob_h(0, ii.begin()) << endl;
  clog << rbm.prob_h(1, io.begin()) << endl;
  clog << rbm.prob_h(0, oi.begin()) << endl;
  clog << rbm.prob_h(1, oo.begin()) << endl;

}