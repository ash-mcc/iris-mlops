#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import pyarrow.feather as feather
import yaml
import pickle
import json
import os
import sys
from dvclive import Live
from matplotlib import pyplot as plt


train_filepath = sys.argv[1]
test_filepath = sys.argv[2]
model_filepath = sys.argv[3]
metrics_filepath = sys.argv[4]
live_dir = sys.argv[5]
importance_filename = sys.argv[6]

live = Live(live_dir)
importance_filepath = os.path.join(live_dir, sys.argv[6]) 


print('Loading model params')

with open("params.yaml", 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


model_args1 =  {"verbose": 1,
                "n_jobs": -1}
model_args2 = params['train']

model_args = {**model_args1, **model_args2}

model = RandomForestClassifier(**model_args)


print('Loading the training & testing dataframes')

train_df = feather.read_feather(train_filepath)
test_df =  feather.read_feather(test_filepath)

print('train_df {}'.format(train_df.head()))
print('test_df {}'.format(test_df.head()))


print('Training the model')

X_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_train = train_df.loc[:, X_cols]
y_train = train_df.species
print('X_train {}, y_train {}'.format(X_train.shape, y_train.shape))

model.fit(X_train, y_train)


print('Writing trained-model to file')

with open(model_filepath, 'wb') as fd:
    pickle.dump(model, fd)


print('Testing the model')

X_test = test_df.loc[:, X_cols]
y_test = test_df.species
print('X_test {}, y_test {}'.format(X_test.shape, y_test.shape))

mean_accuracy =  model.score(X_test, y_test)

y_hat_proba = model.predict_proba(X_test)
print('y_hat_proba {} {}'.format(y_hat_proba.shape, y_hat_proba[:5]))

y_hat = model.predict(X_test)
print('y_hat {} {}'.format(y_hat.shape, y_hat))



print('Writing metrics to file')

with open(metrics_filepath, 'w') as outfile:
    json.dump(
        {"train":
         {"mean_accuracy": mean_accuracy}},
        outfile)
        
fig, axes = plt.subplots(dpi=100)
fig.subplots_adjust(bottom=0.2, top=0.95)
importances = model.feature_importances_
forest_importances = pd.Series(importances, index=X_cols) #.nlargest(n=30)
axes.set_ylabel("Mean decrease in impurity")
forest_importances.plot.bar(ax=axes)
fig.savefig(importance_filepath)


live = Live(live_dir)
# live.log_plot("roc", y_test, y_hat)
# live.log("avg_prec", metrics.average_precision_score(y_test, y_hat))
# live.log("roc_auc", metrics.roc_auc_score(y_test, y_hat))
live.log_plot("confusion_matrix", y_test.squeeze(), y_hat_proba.argmax(-1))


