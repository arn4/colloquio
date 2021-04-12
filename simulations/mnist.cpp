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
const unsigned SAVE_RBM_EVERY = 10;
using real_value = double;
const real_value LEARNING_RATE = 0.05;
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
  BinaryRBM<real_value> rbm_cd10 (PIXELS, HIDDEN_SIZE, rng);
  BinaryRBM<real_value> rbm_pcd10(PIXELS, HIDDEN_SIZE, rng);
  BinaryRBM<real_value> rbm_pcd30(PIXELS, HIDDEN_SIZE, rng);

  CD cd1(rbm_cd1, ts, 1, rng, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM);
  PCD pcd1(rbm_pcd1, ts, 1, rng, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM);
  CD cd10(rbm_cd10, ts, 10, rng, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM);
  PCD pcd10(rbm_pcd10, ts, 10, rng, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM);
  PCD pcd30(rbm_pcd30, ts, 30, rng, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM);

  ofstream result(to_string(seed)+"/psl.txt");
  for (unsigned e = 1; e <= EPOCHS; e++) {
    clog << "Epoch " << e << endl;
    thread epoch_cd1(&CD::epoch, cd1, 0);
    thread epoch_pcd1(&PCD::epoch, pcd1, 0);
    thread epoch_cd10(&CD::epoch, cd10, 0);
    thread epoch_pcd10(&PCD::epoch, pcd10, 0);
    thread epoch_pcd30(&PCD::epoch, pcd30, 0);
    epoch_cd1.join();
    epoch_pcd1.join();
    epoch_cd10.join();
    epoch_pcd10.join();
    epoch_pcd30.join();

    if (e%MONITOR_EVERY==0) {
      auto future_psl_cd1 = async(&CD::log_pseudolikelihood, cd1);
      auto future_psl_pcd1 = async(&PCD::log_pseudolikelihood, pcd1);
      auto future_psl_cd10 = async(&CD::log_pseudolikelihood, cd10);
      auto future_psl_pcd10 = async(&PCD::log_pseudolikelihood, pcd10);
      auto future_psl_pcd30 = async(&PCD::log_pseudolikelihood, pcd30);

      real_value psl_cd1  = future_psl_cd1.get();
      real_value psl_pcd1 = future_psl_pcd1.get();
      real_value psl_cd10  = future_psl_cd10.get();
      real_value psl_pcd10 = future_psl_pcd10.get();
      real_value psl_pcd30 = future_psl_pcd30.get();

      clog << "  PSL CD-1: "  << psl_cd1 << endl;
      clog << "  PSL PCD-1: " << psl_pcd1  << endl;
      clog << "  PSL CD-10: "  << psl_cd10 << endl;
      clog << "  PSL PCD-10: " << psl_pcd10  << endl;
      clog << "  PSL PCD-30: " << psl_pcd30  << endl;

      result << e;
      result << ' ' << psl_cd1;
      result << ' ' << psl_pcd1;
      result << ' ' << psl_cd10;
      result << ' ' << psl_pcd10;
      result << ' ' << psl_pcd30;
      result << endl;

      if (e%SAVE_RBM_EVERY == 0) {
        rbm_cd1.save_on_file(to_string(seed)+"/rbm/cd1_ep"+to_string(e)+".rbm");
        rbm_pcd1.save_on_file(to_string(seed)+"/rbm/pcd1_ep"+to_string(e)+".rbm");
        rbm_cd10.save_on_file(to_string(seed)+"/rbm/cd10_ep"+to_string(e)+".rbm");
        rbm_pcd10.save_on_file(to_string(seed)+"/rbm/pcd10_ep"+to_string(e)+".rbm");
        rbm_pcd30.save_on_file(to_string(seed)+"/rbm/pcd30_ep"+to_string(e)+".rbm");
      }
    }
  }
  result.close();
  clog << "Done!" << endl << endl;
  rbm_cd1.save_on_file(to_string(seed)+"/cd1.rbm");
  rbm_pcd1.save_on_file(to_string(seed)+"/pcd1.rbm");
  rbm_cd10.save_on_file(to_string(seed)+"/cd10.rbm");
  rbm_pcd10.save_on_file(to_string(seed)+"/pcd10.rbm");
  rbm_pcd30.save_on_file(to_string(seed)+"/pcd30.rbm");
}