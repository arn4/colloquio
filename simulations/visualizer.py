import numpy as np
import matplotlib.pyplot as plt

seed = 567137

ROW = 5
COL = 4
IMAGE_HEIGHT = 28
IMAGE_WIDTH  = IMAGE_HEIGHT

IMAGES = ROW * COL
plt.style.use('seaborn')

samples = np.loadtxt(str(seed)+'/samples-cd1.txt')
samples = samples.reshape((IMAGES, IMAGE_WIDTH, IMAGE_HEIGHT))

fig, axs = plt.subplots(ROW, COL)


for i in range(ROW):
  for j in range(COL):
    axs[i,j].imshow(samples[i*COL+j])
    axs[i,j].set(xticklabels=[])
    axs[i,j].set(yticklabels=[])

plt.show()