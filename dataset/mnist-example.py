import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('n',['#D0D0D0','#A00000'])

digits_ = np.loadtxt('mnist-example.txt')
digits = digits_.reshape((10, 28, 28))

fig, axs = plt.subplots(5, 2, figsize=(4,10))

for i in range(5):
  for j in range(2):
    axs[i,j].imshow(digits[i*2+j], cmap=cmap)
    axs[i,j].set(xticklabels=[])
    axs[i,j].set(yticklabels=[])
    axs[i,j].axis('off')

fig.savefig('../latex/img/mnist-example.pdf', format = 'pdf', bbox_inches = 'tight')