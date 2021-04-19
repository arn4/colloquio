import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble='\\usepackage{amsmath} \\usepackage{siunitx}')
plt.style.use('seaborn')

seed = int(sys.argv[1])
trained = ["cd-1", "pcd-1", "mf-3", "tap2-3", "tap3-3", "pmf-3", "ptap2-3", "ptap3-3"]
to_plot = [True,   True,    False,   False,     False,     True,    True,      True     ]


epoch, *_alg = np.loadtxt(str(seed)+'/psl.txt', unpack=True)
alg = np.array(_alg)


fig, ax = plt.subplots(figsize=(7,6))
ax.set_xlabel("Epochs")
ax.set_ylabel("Pseudolikelikelihood $\\left[\\frac{\\mathcal{L}}{\\text{unit}\\cdot\\text{sample}}\\right]$")
for  i in range(len(trained)):
  if to_plot[i]:
    ax.plot(epoch, alg[i], label=trained[i].upper())

ax.legend()
fig.savefig(str(seed)+"/learning-result/psl-plot.pdf", format = 'pdf', bbox_inches = 'tight')