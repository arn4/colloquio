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
#include <ExtendedMeanField.hpp>


using namespace std;
using namespace rbm;

const size_t   DIGITS = 10; // from 0 to 9
const size_t   PIXELS = 28*28;
const size_t   TRAINING_SET_SIZE = 60000;
const unsigned HIDDEN_SIZE = 500;
const unsigned DEFAULT_SEED = 64770;
const unsigned NUM_CONVERGING_MAGNETIZATION = 10;
const unsigned EPOCHS = 50;
const unsigned MONITOR_EVERY = 1;
const unsigned SAVE_RBM_EVERY = 5;
using real_value = double;
const real_value LEARNING_RATE = 0.05;
const real_value WEIGHT_DECAY = .0001;
const real_value MOMENTUM = 0.;

// Insert here the nuber of iterations you would like to do, for each algortihm
const vector<vector<unsigned>> algs = {
  {1,}, // cd
  {1}, // pcd
  {3}, // mf
  {3}, // tap2
  {3}, // tap3
  {3}, // pmf
  {3}, // ptap2
  {3}, // ptap3
};


using CD = ContrastiveDivergence<real_value, PIXELS, DIGITS>;
using PCD = PersistentContrastiveDivergence<real_value, PIXELS, DIGITS>;
using MF = MeanField<real_value, PIXELS, DIGITS>;
using TAP2s = TAP2<real_value, PIXELS, DIGITS>;
using TAP3s = TAP3<real_value, PIXELS, DIGITS>;
using PMF = PersistentMeanField<real_value, PIXELS, DIGITS>;
using PTAP2 = PersistentTAP2<real_value, PIXELS, DIGITS>;
using PTAP3 = PersistentTAP3<real_value, PIXELS, DIGITS>;

const vector <string> alg_names = {"cd", "pcd", "mf", "tap2", "tap3", "pmf", "ptap2", "ptap3"};

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

  vector<vector<BinaryRBM<real_value>>> rbm(alg_names.size());
  for (unsigned s = 0; s < alg_names.size(); s++) {
    rbm[s] = vector<BinaryRBM<real_value>>(algs[s].size(), BinaryRBM<real_value>(PIXELS, HIDDEN_SIZE, rng));
  }
 
  vector<CD> alg_cd;
  for (unsigned k = 0; k< algs[0].size();k++) {
    alg_cd.push_back(CD(rbm[0][k], ts, algs[0][k], rng, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM));
  }
  vector<PCD> alg_pcd;
  for (unsigned k = 0; k< algs[1].size();k++) {
    alg_pcd.push_back(PCD(rbm[1][k], ts, algs[1][k], rng, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM));
  }
  vector<MF> alg_mf;
  for (unsigned k = 0; k< algs[2].size();k++) {
    alg_mf.push_back(MF(rbm[2][k], ts, rng, NUM_CONVERGING_MAGNETIZATION, algs[2][k], LEARNING_RATE, WEIGHT_DECAY, MOMENTUM));
  }
  vector<TAP2s> alg_tap2;
  for (unsigned k = 0; k< algs[3].size();k++) {
    alg_tap2.push_back(TAP2s(rbm[3][k], ts, rng, NUM_CONVERGING_MAGNETIZATION, algs[3][k], LEARNING_RATE, WEIGHT_DECAY, MOMENTUM));
  }
  vector<TAP3s> alg_tap3;
  for (unsigned k = 0; k< algs[4].size();k++) {
    alg_tap3.push_back(TAP3s(rbm[4][k], ts, rng, NUM_CONVERGING_MAGNETIZATION, algs[4][k], LEARNING_RATE, WEIGHT_DECAY, MOMENTUM));
  }
  vector<PMF> alg_pmf;
  for (unsigned k = 0; k< algs[5].size();k++) {
    alg_pmf.push_back(PMF(rbm[5][k], ts, rng, NUM_CONVERGING_MAGNETIZATION, algs[5][k], LEARNING_RATE, WEIGHT_DECAY, MOMENTUM));
  }
  vector<PTAP2> alg_ptap2;
  for (unsigned k = 0; k< algs[6].size();k++) {
    alg_ptap2.push_back(PTAP2(rbm[6][k], ts, rng, NUM_CONVERGING_MAGNETIZATION, algs[6][k], LEARNING_RATE, WEIGHT_DECAY, MOMENTUM));
  }
  vector<PTAP3> alg_ptap3;
  for (unsigned k = 0; k< algs[7].size();k++) {
    alg_ptap3.push_back(PTAP3(rbm[7][k], ts, rng, NUM_CONVERGING_MAGNETIZATION, algs[7][k], LEARNING_RATE, WEIGHT_DECAY, MOMENTUM));
  }

  ofstream result(to_string(seed)+"/psl.txt");
  for (unsigned e = 1; e <= EPOCHS; e++) {
    clog << "Epoch " << e << endl;
    vector<thread> thr;
    for (auto& a: alg_cd) {
      thr.push_back(thread(&CD::epoch, a, 0));
    }
    for (auto& a: alg_pcd) {
      thr.push_back(thread(&PCD::epoch, a, 0));
    }
    for (auto& a: alg_mf) {
      thr.push_back(thread(&MF::epoch, a, 0));
    }
    for (auto& a: alg_tap2) {
      thr.push_back(thread(&TAP2s::epoch, a, 0));
    }
    for (auto& a: alg_tap3) {
      thr.push_back(thread(&TAP3s::epoch, a, 0));
    }
    for (auto& a: alg_pmf) {
      thr.push_back(thread(&PMF::epoch, a, 0));
    }
    for (auto& a: alg_ptap2) {
      thr.push_back(thread(&PTAP2::epoch, a, 0));
    }
    for (auto& a: alg_ptap3) {
      thr.push_back(thread(&PTAP3::epoch, a, 0));
    }

    for (auto& t: thr) {t.join();}

    if (e%MONITOR_EVERY==0) {
      vector<vector<future<real_value>>> fpsl(algs.size());
      for (auto& a: alg_cd) {
        fpsl[0].push_back(async(&CD::log_pseudolikelihood, a));
      }
      for (auto& a: alg_pcd) {
        fpsl[1].push_back(async(&CD::log_pseudolikelihood, a));
      }
      for (auto& a: alg_mf) {
        fpsl[2].push_back(async(&MF::log_pseudolikelihood, a));
      }
      for (auto& a: alg_tap2) {
        fpsl[3].push_back(async(&TAP2s::log_pseudolikelihood, a));
      }
      for (auto& a: alg_tap3) {
        fpsl[4].push_back(async(&TAP3s::log_pseudolikelihood, a));
      }
      for (auto& a: alg_pmf) {
        fpsl[5].push_back(async(&PMF::log_pseudolikelihood, a));
      }
      for (auto& a: alg_ptap2) {
        fpsl[6].push_back(async(&PTAP2::log_pseudolikelihood, a));
      }
      for (auto& a: alg_ptap3) {
        fpsl[7].push_back(async(&PTAP3::log_pseudolikelihood, a));
      }
      
      vector<vector<real_value>> psl(algs.size());

      for (unsigned s = 0; s < alg_names.size(); s++) {
        for (auto& f: fpsl[s]) {
          psl[s].push_back(f.get());
        }
      }

      for (unsigned s = 0; s < algs.size(); s++) {
        for (unsigned f = 0; f < algs[s].size(); f++) {
          clog << "  PSL "<<alg_names[s]<<"-"<<algs[s][f]<<": "<<psl[s][f]<<endl;
        }
      }

      result << e;
      for (unsigned s = 0; s < algs.size(); s++) {
        for (unsigned f = 0; f < algs[s].size(); f++) {
          result << ' ' << psl[s][f];
        }
      }
      result << endl;

      if (e%SAVE_RBM_EVERY == 0) {
        for (unsigned s = 0; s < algs.size(); s++) {
          for (unsigned f = 0; f < algs[s].size(); f++) {
            rbm[s][f].save_on_file(to_string(seed)+"/rbm/"+alg_names[s]+"-"+to_string(algs[s][f])+"_ep"+to_string(e)+".rbm.txt");
          }
        }
      }
    }
  }
  result.close();

  for (unsigned s = 0; s < algs.size(); s++) {
    for (unsigned f = 0; f < algs[s].size(); f++) {
      rbm[s][f].save_on_file(to_string(seed)+"/"+alg_names[s]+"-"+to_string(algs[s][f])+".rbm.txt");
    }
  }
  clog << "Done!" << endl << endl;
}