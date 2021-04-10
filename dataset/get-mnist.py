import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm

OFFSET = 20

mnist_data = tfds.load("mnist")
train, test = mnist_data["train"], mnist_data["test"]

def apply_offset(x):
  if x <= OFFSET:
    return int(0)
  else:
    return int(1)

grayscale2bw = np.vectorize(apply_offset)

def example2myformat(example):
  image, label = example['image'], example['label']
  image = grayscale2bw(image.reshape(((28*28))))
  return str(label) + '\t' + ' '.join(map(str, image))

with open('mnist-test.txt', 'w') as write_test:
  for example in tqdm(tfds.as_numpy(test)):
    write_test.write(example2myformat(example)+'\n')

with open('mnist-train.txt', 'w') as write_train:
  for example in tqdm(tfds.as_numpy(train)):
    write_train.write(example2myformat(example)+'\n')
    