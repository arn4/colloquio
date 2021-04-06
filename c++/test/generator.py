"""
This program generates a training set.
"""
INDIPENDENT_FEATURE_SIZE = 4
FEATURES_SIZE = 4*INDIPENDENT_FEATURE_SIZE
MEAN_ACTIVATION_PROB = 0.3
STD_ACTIVATION_PROB = 0.5/3.

TRAINING_SET_SIZE = 500

import numpy as np

probabilities = np.minimum(1., np.maximum(0., np.random.normal(MEAN_ACTIVATION_PROB, STD_ACTIVATION_PROB, INDIPENDENT_FEATURE_SIZE)))
print(probabilities)
with open('ts.txt', 'w+') as ts_file:
  for i in range(TRAINING_SET_SIZE):
    indip_sample = np.random.binomial(1, probabilities)
    sample = np.concatenate((indip_sample, np.bitwise_and(indip_sample, np.flip(indip_sample)), np.bitwise_or(indip_sample, np.flip(indip_sample)), np.bitwise_xor(indip_sample, np.flip(indip_sample))))
    for b in sample:
      ts_file.write(f'{b} ')
    ts_file.write('\n')
    

