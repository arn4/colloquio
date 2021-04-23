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

//const size_t   DIGITS = 10; // from 0 to 9
const size_t   PIXELS = 28*28;
const unsigned HIDDEN_SIZE = 500;
const unsigned DEFAULT_SEED = 64770;
const unsigned SAMPLES = 20;
const unsigned STEPS_TO_STATIONARY = 50;
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

  
  // Generate TestSet
  clog << "Generation of the samples... " << endl;
  vector<vector<vector<bool>>> samples(trained.size());

  vector<MarkovChain<real_value>> mc;
  for (unsigned r = 0; r < trained.size(); r++) {
    mc.push_back(MarkovChain<real_value>(rbms[r], rng));
  }

  for (unsigned r = 0; r < trained.size(); r++) {
    for (unsigned i = 1; i <= SAMPLES; i++) {
      mc[r].init_random_h();
      mc[r].next_step_v();
      mc[r].evolve(STEPS_TO_STATIONARY);
      auto sample = mc[r].v();
      samples[r].push_back(vector<bool>(sample.begin(), sample.end()));
    }
  }
  clog << "Done!" << endl << endl;
    
  clog << "Writing on file... " << endl;

  for (unsigned r = 0; r < trained.size(); r++) {
    ofstream sout(to_string(seed)+"/generated-set/samples-"+trained[r]+".txt");
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