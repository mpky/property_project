#!/usr/bin/env python
# coding: utf-8

"""
This script applies the the Isolation Forest anomaly detection
algorithm to the property data and then return both visualizations and
performance metrics.
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import make_scorer, precision_score, accuracy_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import os
import yaml
import seaborn as sns
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set_style('dark')

def config_loader():
    """Set config path and load variables."""
    config_path = os.path.join('./configs/config.yaml')
    with open(config_path,'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    return cfg

def read_data():
    """Read in date from true_labels file."""
    cfg = config_loader()
    df = pd.read_hdf(cfg['true_labels'])
    df = df.reset_index()
    X = df.iloc[:,2:-1]
    y = df.crim_prop
    return df, X, y

def output_gen_stats():
    """Load Data and Remove Columns."""
    df, X, y = read_data()
    print('\n'*2)
    print("Number of properties:", len(df))

    # Get criminal property rate
    crim_prop_rate = 1 - (len(df[df['crim_prop']==0]) / len(df))
    print("Criminal property rate: {:.5%}".format(crim_prop_rate))

def relabel_props():
    """
    Re-label the normal properties with 1 and the criminal ones with -1.
    Then, normalize and split into train and test.
    """
    df, X, y = read_data()
    df['binary_y'] = [1 if x==0 else -1 for x in df.crim_prop]

    # Normalize the data
    X_norm = preprocessing.normalize(X)
    y = df.binary_y

    # Split the data into training and test
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(
        X_norm, y, test_size=0.33, random_state=42
    )
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm

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

def metrics_iforest(y_true,y_pred):
    """Return model metrics."""
    print('Recall:',recall_score(
        y_true,
        y_pred,
        zero_division=0,
        pos_label=-1
    ))
    print('Precision:',precision_score(
        y_true,
        y_pred,
        zero_division=0,
        pos_label=-1
    ))

    print("AUC:", roc_auc_score(y_true, y_pred))

def anomaly_plot(anomaly_scores,anomaly_scores_list,title):
    """Plot histograms of anomaly scores."""
    plt.figure(figsize=[7,7])
    plt.subplot(211)
    plt.hist(anomaly_scores,bins=100,log=False,color='royalblue')
    for xc in anomaly_scores_list:
        plt.axvline(x=xc,color='red',linestyle='--',linewidth=0.5,label='criminally-linked property')
    plt.title(title,fontsize=11)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),fontsize=10)
    plt.ylabel('Number of properties',fontsize=10)

    plt.subplot(212)
    plt.hist(anomaly_scores,bins=100,log=True,color='royalblue')
    for xc in anomaly_scores_list:
        plt.axvline(x=xc,color='red',linestyle='--',linewidth=0.5,label='criminally-linked property')
    plt.xlabel('Anomaly score',fontsize=10)
    plt.ylabel('Number of properties',fontsize=10)
    plt.title('{} (Log Scale)'.format(title),fontsize=11)

    plt.show()

def fit_training():
    """Gridsearch to fit model to data."""
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = relabel_props()

    scoring = {
        'AUC': 'roc_auc',
        'Recall': make_scorer(recall_score,pos_label=-1),
        'Precision': make_scorer(precision_score,pos_label=-1)
    }

    gs = GridSearchCV(
        IsolationForest(max_samples=0.25, random_state=42,n_estimators=100),
        param_grid={'contamination': np.arange(0.01, 0.25, 0.05)},
        scoring=scoring,
        refit='Recall',
        verbose=0,
        cv=3
    )

    gs.fit(X_train_norm,y_train_norm)
    return gs

def eval_training():
    """Evaluate model performance on training data."""
    gs = fit_training()
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = relabel_props()

    y_pred_train_gs = gs.predict(X_train_norm)
    print('\n')
    print("Model performance on training data:")
    metrics_iforest(y_train_norm,y_pred_train_gs)

    conf_matrix = confusion_matrix(y_train_norm, y_pred_train_gs)
    print('\n')
    print("Confusion matrix for training data:")
    print(conf_matrix,'\n')
    plot_confusion_matrix(conf_matrix, title='Isolation Forest Confusion Matrix on Training Data')
    print('#'*40,'\n')


def eval_testing():
    """Evaluate model performance on test data."""
    gs = fit_training()
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = relabel_props()

    y_pred_test_gs = gs.predict(X_test_norm)
    print("Model performance on test data:")
    metrics_iforest(y_test_norm,y_pred_test_gs)
    print('\n')

    conf_matrix = confusion_matrix(y_test_norm, y_pred_test_gs)
    print("Confusion matrix for test data:")
    print(conf_matrix,'\n')
    print('#'*40,'\n')
    plot_confusion_matrix(conf_matrix, title='Isolation Forest Confusion Matrix on Test Data')

def plot_anomaly_scores():
    """
    Calculate and visualize Distribution of Anomaly Scores.
    """
    gs = fit_training()
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = relabel_props()

    train_df = pd.DataFrame(X_train_norm)
    y_train_series = y_train_norm.reset_index()
    train_df['y_value'] = y_train_series.binary_y
    train_df['anomaly_scores'] = gs.decision_function(X_train_norm)
    anomaly_scores_list = train_df[train_df.y_value==-1]['anomaly_scores']

    print("Outlier scores for each class in training data:")
    print("Mean score for abnormal:",np.mean(anomaly_scores_list))
    print("Mean score for normal:",np.mean(train_df[train_df.y_value==1]['anomaly_scores']))
    print('\n'*3)

    anomaly_plot(train_df['anomaly_scores'],
                 anomaly_scores_list,
                 title='Distribution of Anomaly Scores across Training Data')

if __name__ == '__main__':
    output_gen_stats()
    eval_training()
    eval_testing()
    plot_anomaly_scores()
