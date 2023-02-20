#!/usr/bin/env python3

from sklearn import metrics
import pandas as pd
import numpy as np
import pyarrow.feather as feather
import pickle
import json
import os
import sys
from dvclive import Live
from matplotlib import pyplot as plt


# ------------------------------------------
# Main fn
#

def main(model_filepath, label_lookup_filepath, test_filepath, metrics_filepath, live_dir, feat_importance_filename, confusion_matrix_name):

    # load the trained model
    with open(model_filepath, "rb") as fd:
        model = pickle.load(fd)
    print(f"Read {model_filepath}")

    # load the label lookup
    with open(label_lookup_filepath, "rb") as fd:
        label_lookup_raw = json.load(fd)
    label_lookup = {int(k): v for k, v in label_lookup_raw.items()}
    print(f"Read {label_lookup_filepath}")

    # load the test data
    test_df =  feather.read_feather(test_filepath)
    print(f"Read {test_filepath}, rows x cols {test_df.shape}")
    print(test_df.head())

    # tailor the data into X,y
    X_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_test = test_df.loc[:, X_cols]
    y_test = test_df.species
    print('X_test, rows x cols {}'.format(X_test.shape))
    print('y_test, rows x cols {}'.format(y_test.shape))

    # score the model on accuracy 
    mean_accuracy =  model.score(X_test, y_test)

    # score the model on other metrics
    y_hat_proba = model.predict_proba(X_test)
    #print('y_hat_proba {} {}'.format(y_hat_proba.shape, y_hat_proba[:5]))
    y_hat = model.predict(X_test)
    #print('y_hat {} {}'.format(y_hat.shape, y_hat))
    cl_rpt = metrics.classification_report(y_test, y_hat, output_dict=True)

    # save the scoring metrics
    with open(metrics_filepath, 'w') as outfile:
        json.dump(
            {"train":
            {"mean_accuracy": mean_accuracy,
            "weighted_avg_f1_score": cl_rpt["weighted avg"]["f1-score"]}},
            outfile)
    print(f"Wrote {metrics_filepath}")

    # get the feature importances from the model
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=X_cols) #.nlargest(n=30)
        
    # prep for DVC Live plots etc.
    live = Live(live_dir)
    images_dir = os.path.join(live_dir, "plots", "images") 
    os.makedirs(images_dir, exist_ok=True)
    importance_filepath = os.path.join(images_dir, feat_importance_filename) 
    
    # plot the feature importances
    fig, ax = plt.subplots(dpi=100)
    fig.subplots_adjust(left=0.2)
    ax.set_xlabel("Mean decrease in impurity")
    #ax.set_title("Feature importances")
    forest_importances.plot.barh(xerr=std, ax=ax, align="center")
    fig.savefig(importance_filepath)
    print(f"Wrote {importance_filepath}")

    # save the confusion matrix
    live.log_sklearn_plot(confusion_matrix_name, y_test.squeeze().map(lambda x: label_lookup[int(x)]), map(lambda x: label_lookup[int(x)], y_hat))
    print(f"Wrote {confusion_matrix_name}")
    

# ------------------------------------------
# For being run as a script
#

if __name__ == "__main__":
    main(*sys.argv[1:])