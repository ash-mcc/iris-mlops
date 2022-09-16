#!/usr/bin/env python3

from sklearn import metrics
import pandas as pd
import numpy as np
import pyarrow.feather as feather
import pickle
import json
import os
import sys
import time
from dvclive import Live
from matplotlib import pyplot as plt


model_filepath = sys.argv[1]
label_lookup_filepath = sys.argv[2]
test_filepath = sys.argv[3]
metrics_filepath = sys.argv[4]
live_dir = sys.argv[5]
importance_filename = sys.argv[6]

live = Live(live_dir)
importance_filepath = os.path.join(live_dir, importance_filename) 


print('Loading the trained model')
with open(model_filepath, "rb") as fd:
    model = pickle.load(fd)


print('Loading label_lookup')
with open(label_lookup_filepath, "rb") as fd:
    label_lookup_raw = json.load(fd)
label_lookup = {int(k): v for k, v in label_lookup_raw.items()}

print('Loading the testing dataframe')

test_df =  feather.read_feather(test_filepath)

print('test_df {}'.format(test_df.head()))


print('Testing the model')

X_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_test = test_df.loc[:, X_cols]
y_test = test_df.species
print('X_test {}, y_test {}'.format(X_test.shape, y_test.shape))


mean_accuracy =  model.score(X_test, y_test)


y_hat_proba = model.predict_proba(X_test)

print('y_hat_proba {} {}'.format(y_hat_proba.shape, y_hat_proba[:5]))

y_hat = model.predict(X_test)
print('y_hat {} {}'.format(y_hat.shape, y_hat))


cl_rpt = metrics.classification_report(y_test, y_hat, output_dict=True)

# use the MCC when a binary classifier
#   sklearn.metrics.matthews_corrcoef(y_true, y_pred, *, sample_weight=None)
# background 
#   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312
#     "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation"
#   https://en.wikipedia.org/wiki/Matthews_correlation_coefficient


print('Writing metrics to file')

with open(metrics_filepath, 'w') as outfile:
    json.dump(
        {"train":
         {"mean_accuracy": mean_accuracy,
          "weighted_avg_f1_score": cl_rpt["weighted avg"]["f1-score"]}},
        outfile)


print('Writing features importances chart to file')

start_time = time.time()
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
forest_importances = pd.Series(importances, index=X_cols) #.nlargest(n=30)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

fig, ax = plt.subplots(dpi=100)
fig.subplots_adjust(left=0.2)
ax.set_xlabel("Mean decrease in impurity")
#ax.set_title("Feature importances")
forest_importances.plot.barh(xerr=std, ax=ax, align="center")
fig.savefig(importance_filepath)


print('Writing confusion matrix to file')

# live.log_plot("roc", y_test, y_hat)#
# live.log("avg_prec", metrics.average_precision_score(y_test, y_hat))
# live.log("roc_
# auc", metrics.roc_auc_score(y_test, y_hat))
live.log_plot("confusion_matrix", y_test.squeeze().map(lambda x: label_lookup[int(x)]), map(lambda x: label_lookup[int(x)], y_hat))
# or?...  live.log_plot("confusion_matrix", y_test.squeeze(), y_hat_proba.argmax(-1))



