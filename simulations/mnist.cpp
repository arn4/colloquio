#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <cstdlib>
#include <future>

#include <BinaryRBM.hpp>
#include <TrainingSet.hpp>
#include <PersistentContrastiveDivergence.hpp>
#include <MarcovChain.hpp>


using namespace std;
using namespace rbm;

const size_t   DIGITS = 10; // from 0 to 9
const size_t   PIXELS = 28*28;
const size_t   TRAINING_SET_SIZE = 60000;
const unsigned HIDDEN_SIZE = 500;
const unsigned DEFAULT_SEED = 64770;
const unsigned EPOCHS = 50;
const unsigned MONITOR_EVERY = 1;
using real_value = double;
const real_value LEARNING_RATE = 0.06;
const real_value WEIGHT_DECAY = 0.0001;
const real_value MOMENTUM = 0.;

using CD = ContrastiveDivergence<real_value, PIXELS, DIGITS>;
using PCD = PersistentContrastiveDivergence<real_value, PIXELS, DIGITS>;

int main(int argc, char *argv[]) {
  unsigned seed;
  if (argc > 1) {
    seed = strtoul(argv[1], nullptr, 10);
  } else {
    seed = DEFAULT_SEED;
  }
  clog << "Using seed = " << seed << endl << endl;
  mt19937 rng(seed);

  // Reading files
  clog << "Reading from file... " << endl;
  vector<vector<vector<bool>>> classes(DIGITS);
  ifstream mnist("./mnist-train.txt");
  for (unsigned i = 1; i <= TRAINING_SET_SIZE; i++) {
    unsigned label;
    mnist >> label;
    vector<bool> tmp(PIXELS);
    for (unsigned q = 0; q < PIXELS; q++) {
      unsigned c;
      mnist >> c;
      tmp[q] = (c==1);
    }
    if(label<DIGITS) {
      classes[label].push_back(tmp);
    }
    //classes[i%DIGITS].push_back(tmp);
  }
  mnist.close();

  clog << "Class sizes: ";
  for (unsigned c = 0; c < DIGITS; c++) {
    clog << classes[c].size() << ' ';
  }
  clog  << endl << endl;

  //Build training set
  clog << "Building Training Set..." << endl;
  TrainingSet<PIXELS, DIGITS> ts(classes, rng);
  clog << "Number of batches: " << ts.num_of_batches() << endl << endl;

  // Training 
  clog << "Training... " << endl;

  BinaryRBM<real_value> rbm_cd1 (PIXELS, HIDDEN_SIZE, rng);
  BinaryRBM<real_value> rbm_pcd1(PIXELS, HIDDEN_SIZE, rng);

  CD cd1(rbm_cd1, ts, 10, rng, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM);
  PCD pcd1(rbm_pcd1, ts, 10, rng, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM);

  ofstream result("mnist-psl-"+to_string(seed)+".txt");
  for (unsigned e = 1; e <= EPOCHS; e++) {
    clog << "Epoch " << e << endl;
    thread epoch_cd1(&CD::epoch, cd1, 0);
    thread epoch_pcd1(&PCD::epoch, pcd1, 0);
    epoch_cd1.join();
    epoch_pcd1.join();
    // pcd1.epoch();
    // real_value psl_pcd1 = pcd1.log_pseudolikelihood();

    if (e%MONITOR_EVERY==0) {
      auto future_psl_cd1 = async(&CD::log_pseudolikelihood, cd1);
      auto future_psl_pcd1 = async(&PCD::log_pseudolikelihood, pcd1);

      real_value psl_cd1  = future_psl_cd1.get();
      real_value psl_pcd1 = future_psl_pcd1.get();

      clog << "  PSL CD-1: "  << psl_cd1 << endl;
      clog << "  PSL PCD-1: " << psl_pcd1  << endl;

      result << e;
      result << ' ' << psl_cd1;
      result << ' ' << psl_pcd1;
      result << endl;
    }
  }
  result.close();
  clog << "Done!" << endl << endl;
  rbm_cd1.save_on_file("mnist-cd1-"+to_string(seed)+".rbm");
  rbm_pcd1.save_on_file("mnist-pcd1-"+to_string(seed)+".rbm");
}