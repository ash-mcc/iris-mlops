#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier
import pyarrow.feather as feather
import pickle
import sys
import dvc.api as dvc


# ------------------------------------------
# Main fn
#

def main(train_filepath, params_filepath, model_filepath):

    # instatiate the untrained model
    all_params = dvc.params_show(params_filepath)
    rf_params = all_params['random_forest']
    model = RandomForestClassifier(**rf_params)

    # load the training data
    train_df = feather.read_feather(train_filepath)
    print(f"Read {train_filepath}, rows x cols {train_df.shape}")
    print(train_df.head())

    # tailor the data into X,y
    X_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_train = train_df.loc[:, X_cols]
    y_train = train_df.species
    print('X_train, rows x cols {}'.format(X_train.shape))
    print('y_train, rows x cols {}'.format(y_train.shape))

    # train the model
    model.fit(X_train, y_train)

    # save the trained model
    with open(model_filepath, 'wb') as fd:
        pickle.dump(model, fd)
    print(f"Wrote {model_filepath}")


# ------------------------------------------
# For being run as a script
#

if __name__ == "__main__":
    main(*sys.argv[1:])
