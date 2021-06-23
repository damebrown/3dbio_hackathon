from clustering import ClusterPipeline
import numpy as np
import matplotlib.pyplot as plt

#####################################
# k_means_experiment:
k_range = (10, 100)
k_step = 10
# k_means_spectral:
eps = [0.1, 0.25, 0.5, 0.75, 0.9]
# Titles:
K_TITLE = "Clustering by correspondence size with {}, k={}"
DBSCAN_TITLE = "Clustering by correspondence size with DBScan, \\u03B5={}"

#####################################


def plot_scatter(X, y, title):
    fig, ax = plt.subplots()
    plt.title(title)
    for g in np.unique(y):
        i = np.where(y == g)
        ax.scatter(X[:, 0][i], X[:, 1][i], label=g)
    plt.show()


def models_with_k_experiment(data: np.array, model: str):
    for k in range(k_range[0], k_range[1], k_step):
        pipe = ClusterPipeline(model, "PCA", "Standard", n_clusters=10)
        pipe.fit(data)
        X = (pipe.pipeline.named_steps)['dim reduction'].named_steps['PCA'].transform(data)
        y = pipe.pipeline.named_steps["clusterer"].named_steps[model].labels_
        plot_scatter(X, y, K_TITLE.format(model, k))


def DBScan_experiment(data):
    for epsilon in eps:
        pipe = ClusterPipeline("DBScan", "PCA", "Standard", eps=epsilon)
        pipe.fit(data)
        X = (pipe.pipeline.named_steps)['dim reduction'].named_steps['PCA'].transform(data)
        y = pipe.pipeline.named_steps["clusterer"].named_steps["Kmeans"].labels_
        plot_scatter(X, y, DBSCAN_TITLE.format(epsilon))


def main():

    # load data into matrix
    # How much data can we load, if not everything how do we split it up in an intelligent way

    # cluster
    # What type of clustering algorithm works best
    # if there's a lot of data can Kmeans / Spectral work? if we reduce the amount of data would they work?
    # PCA or Tsne for dimensional reduction?

    # plot
    # heatmaps vs scatter plots

    np.random.seed(1)
    pipe = ClusterPipeline("Kmeans", "PCA", "Standard", n_clusters=10, max_iter=500)
    rand_data = np.random.random((10000, 10000) )

    Yt = pipe.fit(rand_data)
    Xt = (pipe.pipeline.named_steps)['dim reduction'].named_steps['PCA'].transform(rand_data)
    labels = pipe.pipeline.named_steps["clusterer"].named_steps["Kmeans"].labels_
    plot_scatter(Xt, y, "bla")



if __name__ == '__main__':
    main()