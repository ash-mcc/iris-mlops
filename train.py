#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pyarrow.feather as feather
import yaml
import pickle
import json

print('Loading model params')

with open("params.yaml", 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


args1 =  {"verbose": 1,
          "n_jobs": -1}
args2 = params['train']

model_args = {**args1, **args2}

model = RandomForestClassifier(**model_args)


print('Loading the training & testing dataframes')

train_df = feather.read_feather("train.arrow")
test_df =  feather.read_feather("test.arrow")



print('Training the model')

X_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_train = train_df.loc[:, X_cols]
y_train = train_df.species
print('X_train {}, y_train {}'.format(X_train.shape, y_train.shape))

model.fit(X_train, y_train)


print('Testing the model')

X_test = test_df.loc[:, X_cols]
y_test = test_df.species
print('X_test {}, y_test {}'.format(X_test.shape, y_test.shape))

mean_accuracy =  model.score(X_test, y_test)


print('Writing trained-model to file')

with open('model.pickle', 'wb') as fd:
    pickle.dump(model, fd)

print('Writing test-metrics to file')

with open('eval.json', 'w') as outfile:
    json.dump(
        {"train":
         {"mean_accuracy": mean_accuracy}},
        outfile)