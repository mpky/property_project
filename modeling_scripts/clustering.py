#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
sns.set_style('darkgrid')

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
    return X, y

def plot_reduce_dimensions():
    """Reduce dimensionality and produce a scatter plot."""
    X, y = read_data()
    X_norm = normalize(X)
    # PCA n_componenets=2
    X_pca = PCA(2).fit_transform(X_norm)

    markers = [".",'X']
    sizes = [8, 100]
    plt.figure(figsize=(12,6))

    # cycle through each class
    for i in range(0,2):
        plt.scatter(
            X_pca[y[y==i].index][:,0],
            X_pca[y[y==i].index][:,1],
            s=sizes[i],
            marker=markers[i]
        )
    plt.title('Dataset Reduced to Two Dimensions',fontsize=15)
    plt.legend(('Non-criminal Property','Criminal Property'), loc='upper left',fontsize=13)
    plt.show()

def compare_labels_clusters():
    """Compare predicted labels to clusters."""
    X, y = read_data()
    km = KMeans(
        n_clusters=2,
        init='k-means++',
        random_state=42
    )
    X_norm = normalize(X)
    # PCA n_componenets=2
    X_pca = PCA(2).fit_transform(X_norm)
    y_km = km.fit_predict(X_pca)
    print('Comparing k-means clusters against the data where k=2:')
    print(pd.crosstab(y_km,y),'\n'*2,'#'*40,'\n')

# Reducing the data to two dimensions does seem to reveal some clustering of the criminal properties.
# PCA to higher dimensions
def pca_kmeans():
    """
    Use PCA to reduce the data to higher dimensions to possibly uncover better
    clustering.
    """
    X, y = read_data()
    X_norm = normalize(X)
    cfg = config_loader()

    km = KMeans(
        n_clusters=2,
        init='k-means++',
        random_state=42
    )

    # Pproduce KMeans crosstabs for dataset reduced to n dimensions
    comp_range = range(1,int(cfg['n_componenets'])+1)
    for i in comp_range:
        X_pca = PCA(n_components=i)

        y_km = km.fit_predict(X_pca.fit_transform(X_norm))
        variance_sum = X_pca.explained_variance_.sum()
        print("Crosstab for {} components".format(i),'\n')
        print("Variance explained:",variance_sum)
        print(pd.crosstab(y_km,y),'\n'*2,'#'*40,'\n')

def plot_elbow():
    """
    Produce a distortion plot which would allow for the elbow method to possibly
    uncover true number of clusters.
    """
    X, y = read_data()
    X_norm = normalize(X)

    distortions = []
    for i in range(1, 9):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    random_state=3)
        km.fit(X_norm)
        distortions.append(km.inertia_)

    plt.figure(figsize=(9,6))
    plt.plot(range(1, 9), distortions, marker='o')
    plt.xlabel('Number of clusters',labelpad=10,fontsize=12)
    plt.ylabel('Distortion',labelpad=10,fontsize=12)
    plt.title('Distortion per Number of Clusters',fontsize=16)
    plt.show()

if __name__ == '__main__':
    plot_reduce_dimensions()
    plot_elbow()
    compare_labels_clusters()
    pca_kmeans()
