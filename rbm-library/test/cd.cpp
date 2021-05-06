#include <iostream>
#include <fstream>
#include <random>

#include <BernoulliRBM.hpp>
#include <TrainingSet.hpp>
#include <ContrastiveDivergence.hpp>

using namespace std;
using namespace rbm;

const unsigned FEATURES_SIZE = 16;
const unsigned TRAINING_SET_SIZE = 500;
const unsigned BATCH_SIZE = 10;
const unsigned SEED = 4584;
const unsigned EPOCHS = 500;
using real_value = double;

int main() {
  ifstream ts_in("./ts.txt");
  ofstream ts_out("./ts_check.txt");

  vector<vector<bool>> _ts(TRAINING_SET_SIZE, vector<bool>(FEATURES_SIZE));
  clog << "Loading TS... ";
  for (auto& sample: _ts) {
    for (unsigned i = 0; i < FEATURES_SIZE; i++) {
      unsigned tmp;
      ts_in >> tmp;
      sample[i] = bool(tmp);
    }
  }
  TrainingSet<FEATURES_SIZE, BATCH_SIZE> ts(_ts);
  // Testing  TrainingSet
  for (unsigned b = 0; b < ts.num_of_batches(); b++) {
    for (unsigned k = 0; k < BATCH_SIZE; k++) {
      auto it = ts.batch(b).get_iterator(k);
      for (unsigned i = 0; i < FEATURES_SIZE; i++) {
        ts_out << *(it+i) << ' ';
      }
      ts_out << endl;
    }
    ts_out << endl;
  }
  clog << endl;
  clog << "  Done!" << endl << "Initializing RBM... ";


  mt19937 rng(SEED);
  BernoulliRBM<real_value> rbm(FEATURES_SIZE, FEATURES_SIZE, rng);
  clog << "Done!" << endl << "Initializing CD-k... ";
  ContrastiveDivergence<real_value, FEATURES_SIZE, BATCH_SIZE> cd(rbm, ts, 5, rng, .001, .001, .0);
  clog << "Done!" << endl << "Running... " << endl;

  for (unsigned e = 1; e <= EPOCHS; e++) {
    if (e%10==0) {
      for (unsigned i = 0; i < 5; i++) {
        clog << rbm.b(i) << ' ';
      } clog << endl;
      for (unsigned j = 0; j < 5; j++) {
        clog << rbm.c(j) << ' ';
      } clog << endl;
      for (unsigned j = 0; j < 5; j++) {
        clog << rbm.w(j,j+5) << ' ';
      } clog << endl;
    }
    clog << "  Epoch " << e << ": ";
    cd.epoch();
    clog << cd.log_pseudolikelihood() << endl;
    clog << " ";
    clog << endl;
  }


  return 0;
}