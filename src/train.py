#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier
import pyarrow.feather as feather
import yaml
import pickle
import sys


train_filepath = sys.argv[1]
model_filepath = sys.argv[2]


print('Loading model params')

with open("params.yaml", 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


model_args1 =  {"verbose": 1,
                "n_jobs": -1}
model_args2 = params['train']

model_args = {**model_args1, **model_args2}

model = RandomForestClassifier(**model_args)


print('Loading the training dataframe')

train_df = feather.read_feather(train_filepath)

print('train_df {}'.format(train_df.head()))


print('Training the model')

X_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_train = train_df.loc[:, X_cols]
y_train = train_df.species
print('X_train {}, y_train {}'.format(X_train.shape, y_train.shape))

model.fit(X_train, y_train)


print('Writing trained model to file')

with open(model_filepath, 'wb') as fd:
    pickle.dump(model, fd)