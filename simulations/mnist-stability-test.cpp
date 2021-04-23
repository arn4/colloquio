#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <cstdlib>

#include <BinaryRBM.hpp>
#include <TrainingSet.hpp>
#include <PersistentContrastiveDivergence.hpp>
#include <MarkovChain.hpp>


using namespace std;
using namespace rbm;

const size_t   DIGITS = 10; // from 0 to 9
const size_t   PIXELS = 28*28;
const unsigned HIDDEN_SIZE =500;
const unsigned DEFAULT_SEED = 64770;
const unsigned SAMPLES = 20;
const unsigned STEPS_TO_STATIONARY = 50;
const unsigned TEST_SET_SIZE = 10000;
using real_value = double;
const vector<string> trained = {"cd-1", "pcd-1", "mf-3", "tap2-3", "tap3-3", "pmf-3", "ptap2-3", "ptap3-3"};

int main(int argc, char *argv[]) {
  unsigned seed;
  if (argc > 1) {
    seed = strtoul(argv[1], nullptr, 10);
  } else {
    seed = DEFAULT_SEED;
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

  // Load the test set
  vector<vector<vector<bool>>> classes(DIGITS);
  ifstream mnist("./mnist-test.txt");
  for (unsigned i = 1; i <= TEST_SET_SIZE; i++) {
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
  }

  clog << "Class sizes: ";
  for (unsigned c = 0; c < DIGITS; c++) {
    clog << classes[c].size() << ' ';
  }
  clog  << endl << endl;

  
  // Generate TestSet
  clog << "Evolution of the samples... " << endl;
  vector<vector<vector<bool>>> samples(trained.size());

  vector<MarkovChain<real_value>> mc;
  for (unsigned r = 0; r < trained.size(); r++) {
    mc.push_back(MarkovChain<real_value>(rbms[r], rng));
  }

  for (unsigned r = 0; r < trained.size(); r++) {
    for (unsigned d = 0; d < DIGITS; d++) {
      for (unsigned i = 0; i < SAMPLES/DIGITS; i++) {
        mc[r].set_v(classes[d][i].begin(), classes[d][i].end());
        mc[r].evolve(STEPS_TO_STATIONARY);
        auto sample = mc[r].v();
        samples[r].push_back(vector<bool>(sample.begin(), sample.end()));
      }
    }
  }
  clog << "Done!" << endl << endl;
    
  clog << "Writing on file... " << endl;
  for (unsigned r = 0; r < trained.size(); r++) {
    ofstream sout(to_string(seed)+"/generated-set/stability-"+trained[r]+".txt");
    for (auto s: samples[r]) {
      for (bool b: s) {
        sout << b << ' ';
      }
      sout << endl;
    }
    sout.close();
  }
   
  return 0;
}