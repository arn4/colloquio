#include <ExtendedMeanField.hpp>

// Explicit inistialization: just for testing
namespace rbm {
  template class MeanField<double, 784, 10>;
  template class PersistentMeanField<double, 784, 10>;
  template class TAP2<double, 784, 10>;
  template class PersistentTAP2<double, 784, 10>;
  template class TAP3<double, 784, 10>;
  template class PersistentTAP3<double, 784, 10>;
}
