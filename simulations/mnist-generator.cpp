#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <cstdlib>

#include <BinaryRBM.hpp>
#include <TrainingSet.hpp>
#include <PersistentContrastiveDivergence.hpp>
#include <MarcovChain.hpp>


using namespace std;
using namespace rbm;

//const size_t   DIGITS = 10; // from 0 to 9
const size_t   PIXELS = 28*28;
const unsigned HIDDEN_SIZE =500;
const unsigned DEFAULT_SEED = 64770;
const unsigned SAMPLES = 20;
const unsigned STEPS_TO_STATIONARY = 400;
using real_value = double;

int main(int argc, char *argv[]) {
  unsigned seed;
  if (argc > 1) {
    seed = strtoul(argv[1], nullptr, 10);
  } else {
    seed = DEFAULT_SEED;
  }
  clog << "Using seed = " << seed << endl << endl;
  mt19937 rng(seed);

  // Generate a random RBM
  clog << "Loading the RBM... ";
  BinaryRBM<real_value> rbm_CD1(PIXELS, HIDDEN_SIZE, rng);
  BinaryRBM<real_value> rbm_PCD1(PIXELS, HIDDEN_SIZE, rng);
  
  
  rbm_CD1.load_from_file("mnist-cd1-64770.rbm");
  rbm_PCD1.load_from_file("mnist-pcd1-64770.rbm");
  clog << "Done!" << endl << endl;

  
  // Generate TestSet
  clog << "Generation of the samples... " << endl;
  vector<vector<bool>> samples_CD1;
  vector<vector<bool>> samples_PCD1;
  MarcovChain<real_value> mc_CD1(rbm_PCD1, rng);
  MarcovChain<real_value> mc_PCD1(rbm_PCD1, rng);

  for (unsigned i = 1; i <= SAMPLES; i++) {
    mc_CD1.init_random_v();
    mc_CD1.evolve(STEPS_TO_STATIONARY);
    auto sample = mc_CD1.v();
    samples_CD1.push_back(vector<bool>(sample.begin(), sample.end()));

    mc_PCD1.init_random_v();
    mc_PCD1.evolve(STEPS_TO_STATIONARY);
    sample = mc_PCD1.v();
    samples_PCD1.push_back(vector<bool>(sample.begin(), sample.end()));
  }
  clog << "Done!" << endl << endl;
    
  clog << "Writing on file... " << endl;
  ofstream cd1_out("./samples-CD1.txt");
  ofstream pcd1_out("./samples-PCD1.txt");

  for (auto s: samples_CD1) {
    for (bool b: s) {
      cd1_out << b << ' ';
    }
    cd1_out << endl;
  }
  cd1_out.close();

  for (auto s: samples_PCD1) {
    for (bool b: s) {
      pcd1_out << b << ' ';
    }
    pcd1_out << endl;
  }
  pcd1_out.close();
   
  return 0;
}