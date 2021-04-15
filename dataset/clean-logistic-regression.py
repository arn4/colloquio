import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import seaborn as sns
import pandas as pd

print('Loading Training set...')
train_data = np.loadtxt('mnist-train.txt')
_train_label, *_train_set = train_data.T
train_label = np.array(_train_label)
train_set = np.array(_train_set).T

print('Loading Test set...')
test_data = np.loadtxt('mnist-test.txt')
_test_label, *_test_set = test_data.T
test_label = np.array(_test_label)
test_set = np.array(_test_set).T

print('Fitting...')
logistic_regression = LogisticRegression(max_iter=10000)
logistic_regression.fit(train_set, train_label)

print('Computing the accuracy...')
predictions = logistic_regression.predict(test_set)
score = metrics.accuracy_score(test_label, predictions)
print(score)

print('Plotting confusion matrix...')
confusion_matrix = 100.*metrics.confusion_matrix(test_label, predictions, normalize='true')

df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"], columns = [i for i in "0123456789"])
fig, ax = plt.subplots(figsize=(7,6))
off_diag_mask = np.eye(*confusion_matrix.shape, dtype=bool)
sns.heatmap(df_cm, annot=True, mask = ~off_diag_mask, cmap='Blues', cbar=False, ax=ax, vmin=95, vmax=100)
sns.heatmap(df_cm, annot=True, mask = off_diag_mask,  cmap='OrRd',  cbar=False, ax=ax, vmin=0., vmax=2.)
ax.set_xlabel("Predicted Digit")
ax.set_ylabel("True Digit")
fig.savefig("clean-regression-conf-matrix.pdf", format = 'pdf', bbox_inches = 'tight')

with open('clean-logistic-accuracy.txt','w') as fw:
  fw.write(str(score))
  fw.close()