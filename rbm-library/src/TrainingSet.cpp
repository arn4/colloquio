#include <TrainingSet.hpp>

// Explicit inistialization: just for testing
namespace rbm {
  template class TrainingBatch<784, 10>;
  template class TrainingSet<784,10>;
}
