#include <iostream>
#include <fstream>
#include <random>

#include <BinaryRBM.hpp>
#include <TrainingSet.hpp>
#include <ContrastiveDivergence.hpp>
#include <MarcovChain.hpp>

using namespace std;
using namespace rbm;

const unsigned FEATURES_SIZE = 30;
const unsigned HIDDEN_SIZE = 20;
const unsigned TRAINING_SET_SIZE = 5000;
const unsigned STEPS_TO_STATIONARY = 400;
const unsigned BATCH_SIZE = 10;
const unsigned SEED = 64770;
const unsigned EPOCHS = 500;
const bool     READ_FROM_FILE = false;
const bool     WRITE_ON_FILE = true;
using real_value = double;

int main() {
  mt19937 rng(SEED);
  vector<vector<bool>> samples;

  // Generate a random RBM
  clog << "Generation of the goal RBM... ";
  BinaryRBM<real_value> goal_rbm(FEATURES_SIZE, HIDDEN_SIZE, rng);
  goal_rbm.init_gaussian_b(0., .01);
  goal_rbm.init_gaussian_c(0., .01);
  goal_rbm.init_gaussian_w(0., .01);
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
  }
  // clog << "samples[0][0] = " << samples[0][0] << endl;
  // clog << "samples[0][1] = " << samples[0][1] << endl;
  // clog << "samples[1][0] = " << samples[1][0] << endl;

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

  // Training 
  clog << "Training... " << endl;
  BinaryRBM<real_value> rbm(FEATURES_SIZE, HIDDEN_SIZE, rng);
  ContrastiveDivergence<real_value, FEATURES_SIZE, BATCH_SIZE> cd(rbm, ts, 30, rng, .001, .0, .0);
  for (unsigned e = 1; e <= EPOCHS; e++) {
    // if (e%10==0) {
    //   for (unsigned i = 0; i < 5; i++) {
    //     clog << rbm.b(i) << ' ';
    //   } clog << endl;
    //   for (unsigned j = 0; j < 5; j++) {
    //     clog << rbm.c(j) << ' ';
    //   } clog << endl;
    //   for (unsigned j = 0; j < 5; j++) {
    //     clog << rbm.w(j,j+5) << ' ';
    //   } clog << endl;
    // }
    clog << "  Epoch " << e << ": ";
    cd.epoch();
    //clog << cd.free_energy() << endl;
    clog << " ";
    clog << endl;
  }
  clog << "Done!" << endl << endl;

  // Comparsion
  for (unsigned i = 0; i < min(unsigned(5), FEATURES_SIZE); i++) {
    clog << "b_"<<i << ": " << rbm.b(i) << " " << goal_rbm.b(i) << " " << abs(rbm.b(i)-goal_rbm.b(i))/goal_rbm.b(i) << endl;
  }

  for (unsigned j = 0; j < min(unsigned(5), HIDDEN_SIZE); j++) {
    clog << "c_"<<j << ": " << rbm.c(j) << " " << goal_rbm.c(j) << " " << abs(rbm.c(j)-goal_rbm.c(j))/goal_rbm.c(j) << endl;
  }

  for (unsigned i = 0; i < min(unsigned(2), FEATURES_SIZE); i++) {
    for (unsigned j = 0; j < min(unsigned(3), HIDDEN_SIZE); j++) {
      clog << "w_"<<i<<","<<j << ": " << rbm.w(i,j) << " " << goal_rbm.w(i,j) << " " << abs(rbm.w(i,j)-goal_rbm.w(i,j))/goal_rbm.w(i,j) << endl;
    }
  }

  return 0;
}