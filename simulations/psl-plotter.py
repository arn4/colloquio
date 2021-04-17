import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble='\\usepackage{amsmath} \\usepackage{siunitx}')
plt.style.use('seaborn')

seed = '2204'
trained = ["cd-1", "pcd-1", "mf-5", "mf-10", "tap2-5", "tap2-10", "tap3-5", "tap3-10", "pmf-15", "ptap2-15", "ptap3-30"]


epoch, *_alg = np.loadtxt(str(seed)+'/psl.txt', unpack=True)
alg = np.array(_alg)


fig, ax = plt.subplots(figsize=(7,6))
ax.set_xlabel("Epochs")
ax.set_ylabel("Pseudolikelikelihood $\\left[\\frac{\\mathcal{L}}{\\text{unit}\\cdot\\text{sample}}\\right]$")
for  i in range(len(trained)):
  ax.plot(epoch, alg[i], label=trained[i].upper())

ax.legend()
fig.savefig(str(seed)+"/learning-result/psl-plot.pdf", format = 'pdf', bbox_inches = 'tight')