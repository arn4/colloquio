import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble='\\usepackage{amsmath} \\usepackage{siunitx}')
plt.style.use('seaborn')

seed = '567137'
trained = ['cd1', 'pcd1', 'cd10', 'pcd10','pcd30']


epoch, *_alg = np.loadtxt(str(seed)+'/psl.txt', unpack=True)
alg = np.array(_alg)


fig, ax = plt.subplots(figsize=(7,6))
ax.set_xlabel("Epochs")
ax.set_ylabel("Pseudolikelikelihood $\\left[\\frac{\\mathcal{L}}{\\text{unit}\\cdot\\text{sample}}\\right]$")
for  i in range(len(trained)):
  ax.plot(epoch, alg[i], label=trained[i].upper())

ax.legend()
fig.savefig(str(seed)+"/learning-result/psl-plot.pdf", format = 'pdf', bbox_inches = 'tight')