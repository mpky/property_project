#!/usr/bin/env python
# coding: utf-8

"""
This script uses pseudolabeling to train a semisupervised Gradient Boosting model.
Outputs model metrics and confusion matrix visualization.
"""

import yaml
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
import warnings
warnings.simplefilter('ignore')

from sklearn.svm import SVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, make_scorer, f1_score


def config_loader():
    """Set config path and load variables."""
    config_path = os.path.join('./configs/config.yaml')
    with open(config_path,'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    return cfg

def read_data():
    """Read in date from true_labels file."""
    cfg = config_loader()
    labeled_df = pd.read_hdf(cfg['true_labels'])
    all_df = pd.read_hdf(cfg['processed'])

    return labeled_df, all_df

def plot_confusion_matrix(conf_matrix, title, classes=['criminally-linked', 'normal'], cmap=plt.cm.Oranges):
    """Plot confusion matrix with heatmap and classification statistics."""
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8,8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=15)
    plt.colorbar(pad=.12)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=9)
    plt.yticks(tick_marks, classes, rotation=45, fontsize=9)

    fmt = '.4%'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="top",
                 fontsize=16,
                 color="white" if conf_matrix[i, j] > thresh else "black")
    plt.ylabel('True label',fontsize=11, rotation=0,labelpad=-25)
    plt.xlabel('Predicted label',fontsize=11,labelpad=-25)
    plt.show()

def trim_data():
    """
    Trim to just the data needed for modeling. Trim outlier properties.
    """
    labeled_df, all_df = read_data()
    cfg = config_loader()

    all_df['price_psf'] = all_df['price_psf'].fillna(0)
    percentile = float(cfg['pctile_filter'])
    trim_prop_df = all_df[(all_df.price_psf<all_df.price_psf.quantile(percentile))]

    nan_limit = int(cfg['nan_limit'])
    check_nan = trim_prop_df.isnull().sum()
    variables_list = check_nan[check_nan<nan_limit].index
    variables_list = variables_list[variables_list.isin(trim_prop_df.columns[trim_prop_df.dtypes!='object'])]
    variables_list = variables_list.drop([
        'py_owner_id','py_addr_zip_cass','prop_val_yr','appraised_val',
        'Prior_Mkt_Val','bexar_2015_market_value','bexar_2016_market_value',
        'bexar_2017_market_value','bexar_2018_market_value','owner_zip_code',
        'property_zip','neighborhood_code'
    ])

    sub_df = trim_prop_df[variables_list]
    sub_df = sub_df.dropna()

    unlabeled_df = sub_df[~(sub_df['prop_id'].isin(labeled_df.prop_id))]

    X_labeled = labeled_df.iloc[:,1:-1]
    y_labeled = labeled_df.crim_prop
    X_unlabeled = unlabeled_df.iloc[:,1:-1]

    return X_labeled, y_labeled, X_unlabeled

def split_labeled():
    """Split the data into train and test."""

    X_labeled, y_labeled, X_unlabeled = trim_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.25, random_state=3
    )

    return X_train, X_test, y_train, y_test

def train_gbc():
    """Train first model on labeled data."""

    print("Loading data.")
    X_labeled, y_labeled, X_unlabeled = trim_data()
    X_train, X_test, y_train, y_test = split_labeled()
    # Train model
    print("Training first model.")
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train, y_train)
    return gbc

def get_pseudo():
    """
    Predict pseudo-labels on unlabeled data and combine with labeled data.
    """

    gbc = train_gbc()
    X_labeled, y_labeled, X_unlabeled = trim_data()

    print("Producing pseudolabels.")
    pseudo_labels = gbc.predict(X_unlabeled)

    # Add pseudo-labels to test
    augmented_test = X_unlabeled.copy(deep=True)
    augmented_test['crim_prop'] = pseudo_labels

    # Take a fraction of the pseudo-labeled data to combine with the labeled training data
    sampled_test = augmented_test.sample(frac=.4,random_state=42)
    print('Length of pseudo-labeled data:',len(sampled_test))

    # Re-merge
    temp_labeled = pd.concat([X_labeled,y_labeled],axis=1)
    # Concat labeled data with pseudo-labeled data
    augmented_labeled = pd.concat([sampled_test,temp_labeled])
    return augmented_labeled

def train_pseudo_gbc():
    """Train new Gradient Boost model on the expanded dataset."""
    augmented_labeled = get_pseudo()


    print("Training second model on pseudolabeled data.")
    gbc_aug = GradientBoostingClassifier(random_state=42)
    gbc_aug.fit(augmented_labeled.iloc[:,:-1], augmented_labeled.crim_prop)
    y_pred_train = gbc_aug.predict(augmented_labeled.iloc[:,:-1])
    return augmented_labeled, y_pred_train

def eval_model():
    """
    Evaluate model performance and visualize confusion matrix.
    """

    augmented_labeled, y_pred_train = train_pseudo_gbc()

    print('\n')
    print("Confusion matrix:")
    conf_matrix = confusion_matrix(augmented_labeled.crim_prop,y_pred_train,labels=[1,0])
    print(conf_matrix,'\n'*2)
    print("Performance Metrics:")
    print('Recall:',recall_score(augmented_labeled.crim_prop,y_pred_train))
    print('Precision:',precision_score(augmented_labeled.crim_prop,y_pred_train))
    print('F1 Score:',f1_score(augmented_labeled.crim_prop,y_pred_train))

    print('\n')

    input_key = input(
        """Enter 'yes' to display confusion matrix. Type anything else to skip: """
    )
    if input_key != 'yes':
        print("Skipping plot. Goodbye.")
        return

    plot_confusion_matrix(conf_matrix, title='Confusion Matrix for Gradient Boosting')

if __name__ == '__main__':
    eval_model()
