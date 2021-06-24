from cluster_pipeline import ClusterPipeline
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import sys
from utils import *
from mpl_toolkits.mplot3d import Axes3D
import pickle

# ------------ Constants --------
# models_with_k_experiment:
k_range = (3, 300)
# k_range = (1, 3)
k_step = 5
# DBScan_experiment:
# eps = [20,]
eps = [0.1, 0.25, 0.5, 0.75, 0.9]
# Titles:
K_TITLE = "Clustering by correspondence size with {}, k={}, {}-{}, {}"
DBSCAN_TITLE = "Clustering by correspondence size with DBScan, \u03B5={}, {}-{},\n {}"
DEFAULT_FIG_PATH = "/cs/usr/linoytsaban_14/PycharmProjects/3dbio_hackathon/data_files/Figures/representatives_1000_{}_{}_{}_{}_{}"
GRAPH_FIG_PATH = "/cs/usr/linoytsaban_14/PycharmProjects/3dbio_hackathon/data_files/Figures/representatives_1000_{}_{}_{}_{}"
GRAPH_TITLE = "Kmeans scores, {}-{}, {}"
PKL_Path = "/cs/usr/linoytsaban_14/PycharmProjects/3dbio_hackathon/data_files/Figures/Pickle/representatives_1000_{}_{}_{}_{}_{}.pickle"
# General constants:
dim_reduction = ("Tsne", 2)
# dim_reduction = (Tsne, 2)
SAVE = True

SCALER = "MinMax"


# --------------------------------


def plot_scatter_2d(X: np.array, y: np.array, title: str, save_path: str = None):
    fig, ax = plt.subplots()
    plt.title(title)
    for marker in np.unique(y):
        i = np.where(y == marker)
        ax.scatter(X[:, 0][i], X[:, 1][i], label=marker)
    if SAVE:
        plt.savefig(save_path)
    plt.show()


def plot_scatter_3d(X: np.array, y: np.array, title: str, save_path: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    for marker in np.unique(y):
        i = np.where(y == marker)
        ax.scatter(X[:, 0][i], X[:, 1][i], X[:, 2][i], label=marker)
    if SAVE:
        plt.savefig(save_path)
    plt.show()


def plot(X: np.array, y: np.array, title: str, save_path: str = None, heatmap: bool = False):
    plt.rcParams.update({'font.size': 8})
    if dim_reduction[1] == 3:
        plot_scatter_3d(X, y, title, save_path)
    else:
        plot_scatter_2d(X, y, title, save_path)
    if heatmap:
        save_path = save_path + "_heatmap" if save_path else save_path
        plot_heatmap(X, y, title, save_path)


def plot_heatmap(X: np.array, y: np.array, title: str, save_path: str = None):
    indices = np.argsort(y, kind='mergesort')  # we need a stable sort
    plt.figure()
    ax = sbn.heatmap(X[indices], cmap="YlGnBu")
    plt.title(title)
    if SAVE:
        plt.savefig(save_path)
    plt.show()


def plot_score_graph(X: np.array, y: np.array, title: str, save_path: str = None):
    y = np.log1p(y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    ax.set_ylabel("K-Means Score")
    ax.set_xlabel("Number of clusters")
    plt.plot(X, y)
    if SAVE:
        plt.savefig(save_path)
    plt.show()


def save_labels(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def models_with_k_experiment(data: np.array, model: str):
    for k in range(k_range[0], k_range[1], k_step):
        pipe = ClusterPipeline(model, dim_reduction, "Standard", n_clusters=k)
        pipe.fit_transform(data)
        X = (pipe.pipeline.named_steps)['dim reduction'].named_steps[dim_reduction[0]].transform(data)
        y = pipe.pipeline.named_steps["clusterer"].named_steps[model].labels_
        plot(X, y, K_TITLE.format(model, k, dim_reduction[0], dim_reduction[1], SCALER), DEFAULT_FIG_PATH.format("Kmeans", k, dim_reduction[0], dim_reduction[1], SCALER))


def DBScan_experiment(data: np.array):
    for epsilon in eps:
        pipe = ClusterPipeline("DBScan", dim_reduction, "MinMax", eps=epsilon)
        pipe.fit(data)
        X = (pipe.pipeline.named_steps)['dim reduction'].named_steps[dim_reduction[0]].transform(data)
        y = pipe.pipeline.named_steps["clusterer"].named_steps["DBScan"].labels_
        num_of_clusters = y.max()
        print(num_of_clusters)
        plot(X, y, DBSCAN_TITLE.format(epsilon, dim_reduction[0], dim_reduction[1], SCALER), DEFAULT_FIG_PATH.format("DBscan", str(epsilon).replace('.', " "), dim_reduction[0], dim_reduction[1], SCALER))


def mini_batch_experiment(data: np.array):
    scores = list()
    for k in range(k_range[0], k_range[1], k_step):
        pipe = ClusterPipeline("MiniBatchKmeans", dim_reduction, "Standard", n_clusters=k, batch_size=10)
        pipe.fit_transform(data)
        if dim_reduction[0]:
            X = (pipe.pipeline.named_steps)['dim reduction'].named_steps[dim_reduction[0]].transform(data)
        else:
            X = data
        y = pipe.pipeline.named_steps["clusterer"].named_steps["MiniBatchKmeans"].labels_
        plot(X, y, K_TITLE.format("MiniBatchKmeans", k, dim_reduction[0], dim_reduction[1], SCALER), DEFAULT_FIG_PATH.format("MinibatchKmeans", k, dim_reduction[0], dim_reduction[1], SCALER), heatmap=False)
        scores.append(pipe.pipeline.named_steps["clusterer"].named_steps["MiniBatchKmeans"].inertia_)
        save_labels(PKL_Path.format("MinibatchKmeans", k, dim_reduction[0], dim_reduction[1], SCALER), y)
    plot_score_graph(list(range(k_range[0], k_range[1], k_step)), scores, GRAPH_TITLE.format(dim_reduction[0], dim_reduction[1], SCALER), GRAPH_FIG_PATH.format("Kmeans", dim_reduction[0], dim_reduction[1], SCALER))


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data', metavar='data', type=str,
                        help='path to dataframe')
    return parser


def load_data(data_path):
    data = triangle_to_symmetric_matrix(data_path)
    return data


def main():
    # load data into matrix
    # How much data can we load, if not everything how do we split it up in an intelligent way

    # cluster
    # What type of clustering algorithm works best
    # if there's a lot of data can Kmeans / Spectral work? if we reduce the amount of data would they work?
    # PCA or Tsne for dimensional reduction?

    # plot
    # heatmaps vs scatter plots

    args = get_args()
    params = args.parse_args(sys.argv[1:])
    data_path = params.data
    data = load_data(data_path)
    mini_batch_experiment(data)
    # DBScan_experiment(data)


if __name__ == '__main__':
    # path = "/cs/usr/punims/punims/Bio3D_Hackathon/3dbio_hackathon/data_files/Figures/Pickle/plot_random_1000_{}_{}_{}_{}_{}.pickle"
    # file = open(path,'rb')
    # object_file = pickle.load(file)
    # print(object_file)
    main()
    # # save_labels(["T", "t"])
