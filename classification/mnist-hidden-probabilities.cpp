#include <iostream>
#include <fstream>
#include <random>
#include <cstdlib>
#include <iomanip>
#include <cassert>

#include <BinaryRBM.hpp>


using namespace std;
using namespace rbm;

const size_t   DIGITS = 10;
const size_t   PIXELS = 28*28;
const unsigned HIDDEN_SIZE = 500;
const unsigned TEST_SET_SIZE = 10000;
const size_t   TRAIN_SET_SIZE = 60000;
//const unsigned EPOCHS = 50;
//const unsigned RBM_EVERY = 10;
using real_value = double;
const vector<string> trained = {"cd-1", "pcd-1", "mf-3", "pmf-3", "tap2-3", "tap3-3", "pmf-3", "ptap2-3", "ptap3-3"};

int main(int argc, char *argv[]) {
  unsigned seed = 0;
  if (argc > 1) {
    seed = strtoul(argv[1], nullptr, 10);
  } else {
    clog << "Specify the trained machines" << endl;
  }
  clog << "Using seed = " << seed << endl << endl;
  mt19937 rng(seed);

  // Load RBM
  clog << "Loading the RBM... ";
  vector<BinaryRBM<real_value>> rbms(trained.size(), BinaryRBM<real_value>(PIXELS, HIDDEN_SIZE, rng));
  
  for (unsigned r = 0; r < trained.size(); r++) {
    rbms[r].load_from_file(to_string(seed)+"/"+trained[r]+".rbm.txt");
  }
  clog << "Done!" << endl << endl;

  // Process the train set
  clog << "Loading the train set...";
  vector<vector<bool>> train_set;
  vector<unsigned> train_label;
  ifstream mnist_train("./mnist-train.txt");
  for (unsigned i = 1; i <= TRAIN_SET_SIZE; i++) {
    unsigned label;
    mnist_train >> label;
    vector<bool> tmp(PIXELS);
    for (unsigned q = 0; q < PIXELS; q++) {
      unsigned c;
      mnist_train >> c;
      tmp[q] = (c==1);
    }
    train_label.push_back(label);
    train_set.push_back(tmp);
  }
  clog  << endl << endl;

  // Load the test set
  clog << "Loading the test set...";
  vector<vector<bool>> test_set;
  vector<unsigned> test_label;
  ifstream mnist_test("./mnist-test.txt");
  for (unsigned i = 1; i <= TEST_SET_SIZE; i++) {
    unsigned label;
    mnist_test >> label;
    vector<bool> tmp(PIXELS);
    for (unsigned q = 0; q < PIXELS; q++) {
      unsigned c;
      mnist_test >> c;
      tmp[q] = (c==1);
    }
    test_label.push_back(label);
    test_set.push_back(tmp);
  }
  clog << endl << endl;

  //Write labels
  ofstream train_label_out(to_string(seed)+"/hidden-magnetization/train-label.txt");
  for (unsigned u: train_label) {
    train_label_out << u << endl;
  }

  ofstream test_label_out(to_string(seed)+"/hidden-magnetization/test-label.txt");
  for (unsigned u: test_label) {
    test_label_out << u << endl;
  }

  
  // Compute Magnetizations
  clog << "Computing magnetization... " << endl;

  for (unsigned r = 0; r < trained.size(); r++) {
    clog << r << '/' << trained.size() << endl;
    // Train
    ofstream train_magn_out(to_string(seed)+"/hidden-magnetization/train-"+trained[r]+".txt");
    train_magn_out << fixed << setprecision(10);
    for (auto& v: train_set) {
      auto prob_h = rbms[r].vec_prob_h(v.begin());
      for (real_value mhj: prob_h) {
        train_magn_out << mhj << ' ';
      }
      train_magn_out << endl;
    }
    train_magn_out.close();
    // Test
    ofstream test_magn_out(to_string(seed)+"/hidden-magnetization/test-"+trained[r]+".txt");
    test_magn_out << fixed << setprecision(10);
    for (auto& v: test_set) {
      auto prob_h = rbms[r].vec_prob_h(v.begin());
      for (real_value mhj: prob_h) {
        test_magn_out << mhj << ' ';
      }
      test_magn_out << endl;
    }
    test_magn_out.close();
  }
  clog << trained.size() << '/' << trained.size() << endl;

  return 0;
}