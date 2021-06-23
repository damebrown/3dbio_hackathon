from clustering import ClusterPipeline
import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(X, y):
    fig, ax = plt.subplots()
    for g in np.unique(y):
        i = np.where(y == g)
        ax.scatter(X[:, 0][i], X[:, 1][i], label=g)
    plt.show()


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
    plot_scatter(Xt, y)



if __name__ == '__main__':
    main()