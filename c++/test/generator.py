"""
This program generates a training set.
"""

FEATURES_SIZE = 20
MEAN_ACTIVATION_PROB = 0.3
STD_ACTIVATION_PROB = 0.5/3.

TRAINING_SET_SIZE = 500

import numpy as np

probabilities = np.minimum(1., np.maximum(0., np.random.normal(MEAN_ACTIVATION_PROB, STD_ACTIVATION_PROB, FEATURES_SIZE)))
print(probabilities)
with open('ts.txt', 'w+') as ts_file:
  for i in range(TRAINING_SET_SIZE):
    sample = np.random.binomial(1, probabilities)
    for b in sample:
      ts_file.write(f'{b} ')
    ts_file.write('\n')
    

