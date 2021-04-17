#include <iostream>
#include <fstream>
#include <random>
#include <thread>

#include <BinaryRBM.hpp>
#include <TrainingSet.hpp>
#include <PersistentContrastiveDivergence.hpp>
#include <MarcovChain.hpp>
#include <ExtendedMeanField.hpp>

using namespace std;
using namespace rbm;

const unsigned FEATURES_SIZE = 150;
const unsigned HIDDEN_SIZE = 80;
const unsigned TRAINING_SET_SIZE = 3000;
const unsigned STEPS_TO_STATIONARY = 40;
const unsigned BATCH_SIZE = 10;
const unsigned SEED = 64770;
const unsigned EPOCHS = 200;
const unsigned K = 3;
const bool     READ_FROM_FILE = false;
const bool     WRITE_ON_FILE = false;
using real_value = double;
// using LearningAlgorithm1 = ContrastiveDivergence<real_value, FEATURES_SIZE, BATCH_SIZE>;
// using LearningAlgorithm2 = PersistentContrastiveDivergence<real_value, FEATURES_SIZE, BATCH_SIZE>;
using LearningAlgorithm1 = ContrastiveDivergence<real_value, FEATURES_SIZE, BATCH_SIZE>;
using LearningAlgorithm2 = PersistentMeanField<real_value, FEATURES_SIZE, BATCH_SIZE>;
const real_value LEARNING_RATE = 0.05;
const real_value WEIGHT_DECAY = 0.00001;
const real_value MOMENTUM = 0.5;

int main() {
  mt19937 rng(SEED);
  vector<vector<bool>> samples;

  // Generate a random RBM
  clog << "Generation of the goal RBM... ";
  BinaryRBM<real_value> goal_rbm(FEATURES_SIZE, HIDDEN_SIZE, rng);
  goal_rbm.init_gaussian_b(0., 0.1);
  goal_rbm.init_gaussian_c(0., 0.1);
  goal_rbm.init_gaussian_w(0., 0.1);
  goal_rbm.save_on_file("generated-goal.rbm");
  clog << "Done!" << endl << endl;

  if (not READ_FROM_FILE) {
    // Generate TestSet
    clog << "Generation of the samples... " << endl;
    MarcovChain<real_value> mc(goal_rbm, rng);
    clog << " |                    |" << endl << "  ";
    for (unsigned i = 1; i <= TRAINING_SET_SIZE; i++) {
      if (i % (TRAINING_SET_SIZE/20)==0) {
        clog << '-';
      }
      mc.init_random_v();
      mc.evolve(STEPS_TO_STATIONARY);
      auto sample = mc.v();
      samples.push_back(vector<bool>(sample.begin(), sample.end()));
    }
    clog << endl;
    if (WRITE_ON_FILE) {
      clog << "Writing on file... " << endl;
      ofstream ts_out("./generated-ts.txt");
      for (auto s: samples) {
        for (bool b: s) {
          ts_out << b << ' ';
        }
        ts_out << endl;
      }
      ts_out.close();
    }
  } else {
    clog << "Reading from file... " << endl;
    ifstream ts_in("./generated-ts.txt");
    for (unsigned i = 1; i <= TRAINING_SET_SIZE; i++) {
      vector<bool> tmp(FEATURES_SIZE, 0);
      for (unsigned q = 0; q < FEATURES_SIZE; q++) {
        int c;
        ts_in >> c;
        tmp[q] = (c==1);
      }
      samples.push_back(tmp);
    }
    ts_in.close();
  }
 

  TrainingSet<FEATURES_SIZE, BATCH_SIZE> ts(samples);
  clog << endl << "Done!" << endl << endl;

  // Testing  TrainingSet
  ofstream test_ts_out("./test_generated-ts.txt");
  for (unsigned b = 0; b < ts.num_of_batches(); b++) {
    for (unsigned k = 0; k < BATCH_SIZE; k++) {
      auto it = ts.batch(b).get_iterator(k);
      for (unsigned i = 0; i < FEATURES_SIZE; i++) {
        test_ts_out << *(it+i) << ' ';
      }
      test_ts_out << endl;
    }
    test_ts_out << endl;
  }
  test_ts_out.close();

  // Training 
  clog << "Training... " << endl;
  BinaryRBM<real_value> rbm1(FEATURES_SIZE, HIDDEN_SIZE, rng);
  BinaryRBM<real_value> rbm2(rbm1.b(), rbm1.c(), rbm1.w(), rng);
  LearningAlgorithm1 la1(rbm1, ts, K, rng, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM);
  LearningAlgorithm2 la2(rbm2, ts, rng, 10, K, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM);
  clog << "Initial PLH1: " << la1.log_pseudolikelihood() << endl;
  clog << "Initial PLH2: " << la2.log_pseudolikelihood() << endl;
  ofstream result("psl.txt");
  for (unsigned e = 1; e <= EPOCHS; e++) {
    clog << "  Epoch " << e << ": " << endl;
    thread st(&LearningAlgorithm1::epoch, la1, 0);
    thread nd(&LearningAlgorithm2::epoch, la2, 0);
    // la1.epoch();
    // la2.epoch();
    st.join();
    real_value psl1 = la1.log_pseudolikelihood();
    clog << psl1 << endl;

    nd.join();
    real_value psl2 = la2.log_pseudolikelihood();
    clog << psl2 << endl;
    result << e << ' ' << psl1 << ' ' << psl2 << endl;
  }
  clog << "Done!" << endl << endl;
  rbm1.save_on_file("learned-on-generated-1.rbm");
  rbm2.save_on_file("learned-on-generated-2.rbm");
  

  return 0;
}