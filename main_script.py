from cluster_pipeline import ClusterPipeline
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# ------------ Constants --------
# models_with_k_experiment:
k_range = (3, 14)
k_step = 1
# DBScan_experiment:
eps = [0.1, 0.25, 0.5, 0.75, 0.9]
# Titles:
K_TITLE = "Clustering by correspondence size with {}, k={}"
DBSCAN_TITLE = "Clustering by correspondence size with DBScan, \u03B5={}"
# General constants:
dim_reduction = ("Tsne", 2)


# --------------------------------


def plot_scatter_2d(X: np.array, y: np.array, title: str):
    fig, ax = plt.subplots()
    plt.title(title)
    for marker in np.unique(y):
        i = np.where(y == marker)
        ax.scatter(X[:, 0][i], X[:, 1][i], label=marker)
    plt.show()


def plot_scatter_3d(X: np.array, y: np.array, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    for marker in np.unique(y):
        i = np.where(y == marker)
        ax.scatter(X[:, 0][i], X[:, 1][i], X[:, 2][i], label=marker)
    plt.show()


def plot(X: np.array, y: np.array, title: str, heatmap: bool = False):
    if dim_reduction[1] == 3:
        plot_scatter_3d(X, y, title)
    else:
        plot_scatter_2d(X, y, title)
    if heatmap:
        plot_heatmap(X, y, title)


def plot_heatmap(X: np.array, y: np.array, title: str):
    indices = np.argsort(y, kind='mergesort')  # we need a stable sort
    plt.figure()
    ax = sbn.heatmap(X[indices], cmap="YlGnBu")
    plt.title(title)
    plt.show()


def models_with_k_experiment(data: np.array, model: str):
    for k in range(k_range[0], k_range[1], k_step):
        pipe = ClusterPipeline(model, dim_reduction, "Standard", n_clusters=k)
        pipe.fit_transform(data)
        X = (pipe.pipeline.named_steps)['dim reduction'].named_steps[dim_reduction[0]].transform(data)
        y = pipe.pipeline.named_steps["clusterer"].named_steps[model].labels_
        plot(X, y, K_TITLE.format(model, k))


def DBScan_experiment(data: np.array):
    for epsilon in eps:
        pipe = ClusterPipeline("DBScan", dim_reduction, "Standard", eps=epsilon)
        pipe.fit(data)
        X = (pipe.pipeline.named_steps)['dim reduction'].named_steps[dim_reduction[0]].transform(data)
        y = pipe.pipeline.named_steps["clusterer"].named_steps["DBScan"].labels_
        plot(X, y, DBSCAN_TITLE.format(epsilon))


def mini_batch_experiment(data: np.array):
    for k in range(k_range[0], k_range[1], k_step):
        pipe = ClusterPipeline("MiniBatchKmeans", dim_reduction, "Standard", n_clusters=k, batch_size=10)
        pipe.fit_transform(data)
        if dim_reduction:
            X = (pipe.pipeline.named_steps)['dim reduction'].named_steps[dim_reduction[0]].transform(data)
        else:
            X = data
        y = pipe.pipeline.named_steps["clusterer"].named_steps["MiniBatchKmeans"].labels_
        plot(X, y, K_TITLE.format("MiniBatchKmeans", k))


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data', metavar='data', type=str,
                        help='path to dataframe')
    return parser


def main():
    # load data into matrix
    # How much data can we load, if not everything how do we split it up in an intelligent way

    # cluster
    # What type of clustering algorithm works best
    # if there's a lot of data can Kmeans / Spectral work? if we reduce the amount of data would they work?
    # PCA or Tsne for dimensional reduction?

    # plot
    # heatmaps vs scatter plots

    import sys
    args = get_args()
    params = args.parse_args(sys.argv[1:])
    data = params.data
    if data:
        our_data = pd.read_csv(data, header=None)
        our_data = our_data[our_data.columns[:-1]]
    else:
        np.random.seed(1)
        our_data = np.random.random((100, 100))

    mini_batch_experiment(our_data)


if __name__ == '__main__':
    main()
