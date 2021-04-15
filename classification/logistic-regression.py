
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

trained = ['cd1', 'pcd1', 'cd10', 'pcd10','pcd30']
seed = 567137

CLEAN_LOGIST_ACCURACY = 0.9182

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble='\\usepackage{amsmath} \\usepackage{siuitx}')

def regression(alg_name, train_label, test_label):
  print(f'Processing {alg_name}')
  print('  Loading Training set...')
  train_set = np.loadtxt(str(seed)+"/hidden-magnetization/train-"+alg_name+".txt")
  print('  Loading Test set...')
  test_set = np.loadtxt(str(seed)+"/hidden-magnetization/test-"+alg_name+".txt")

  print('  Fitting...')
  logistic_regression = LogisticRegression(max_iter=10000)
  logistic_regression.fit(train_set, train_label)

  print('  Computing the accuracy...')
  predictions = logistic_regression.predict(test_set)
  score = metrics.accuracy_score(test_label, predictions)
  print(f' Score: {score}')
  print('  Plotting confusion matrix...')
  confusion_matrix = 100*metrics.confusion_matrix(test_label, predictions, normalize='true')
  df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"], columns = [i for i in "0123456789"])

  fig, ax = plt.subplots(figsize=(7,6))
  off_diag_mask = np.eye(*confusion_matrix.shape, dtype=bool)
  sns.heatmap(df_cm, annot=True, mask = ~off_diag_mask, cmap='Blues', cbar=False, ax=ax, vmin=95, vmax=100)
  sns.heatmap(df_cm, annot=True, mask = off_diag_mask,  cmap='OrRd',  cbar=False, ax=ax, vmin=0., vmax=2.)
  ax.set_xlabel("Predicted Digit")
  ax.set_ylabel("True Digit")
  fig.savefig(str(seed)+"/classification-result/"+alg_name+"conf-matrix.pdf", format = 'pdf', bbox_inches = 'tight')
  
  return score

train_label = np.loadtxt(str(seed)+"/hidden-magnetization/train-label.txt")
test_label = np.loadtxt(str(seed)+"/hidden-magnetization/test-label.txt")

accuracies = np.array([regression(alg_name, train_label, test_label) for alg_name in trained])

np.savetxt(str(seed)+"/classification-result/accuracies.txt", accuracies)

accuracies = np.loadtxt(str(seed)+"/classification-result/accuracies.txt")

plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.bar(list(map(str.upper, trained)), 100*accuracies, color=sns.color_palette("pastel")[:len(trained)])
ax.axhline(100*CLEAN_LOGIST_ACCURACY, ls=':', c='darkslategray')
ax.set_ylim(90., 100.)
ax.set_xlabel("Training Algorithm")
ax.set_ylabel("Accuracy $\\left[\\si{\\percent}\\right]$")
fig.savefig(str(seed)+"/classification-result/acc-hist.pdf", format = 'pdf', bbox_inches = 'tight')


