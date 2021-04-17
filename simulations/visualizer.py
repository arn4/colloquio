import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

seed = 567137
trained = ["cd1", "pcd1", "cd10", "pcd10", "pcd30"]

# seed = '2204'
# trained = ["cd-1", "pcd-1", "mf-5", "mf-10", "tap2-5", "tap2-10", "tap3-5", "tap3-10", "pmf-15", "ptap2-15", "ptap3-30"]

ROW = {'stability':2, 'samples':5}
COL = {'stability':10, 'samples':4}

IMAGE_HEIGHT = 28
IMAGE_WIDTH  = IMAGE_HEIGHT

IMAGES = ROW['samples'] * COL['samples']
assert(IMAGES == ROW['stability'] * COL['stability'])

plt.style.use('seaborn')

def plot_set(stype, alg_name):
  samples = np.loadtxt(str(seed)+'/generated-set/'+stype+'-'+alg_name+'.txt')
  samples = samples.reshape((IMAGES, IMAGE_WIDTH, IMAGE_HEIGHT))

  fig, axs = plt.subplots(ROW[stype], COL[stype], figsize=(1.5*COL[stype],1.5*ROW[stype]))

  for j in range(COL[stype]):
    for i in range(ROW[stype]):
      axs[i,j].imshow(samples[j*ROW[stype]+i], cmap=LinearSegmentedColormap.from_list('n',['#D0D0D0','#A00000']))
      axs[i,j].set(xticklabels=[])
      axs[i,j].set(yticklabels=[])
      axs[i,j].axis('off')
  fig.savefig(str(seed)+'/learning-result/'+stype+'-'+alg_name+'.pdf', format = 'pdf', bbox_inches = 'tight')

for alg_name in trained:
  plot_set('samples', alg_name)
  plot_set('stability', alg_name)
