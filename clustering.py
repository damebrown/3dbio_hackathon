import matplotlib.pyplot as plt
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np


import matplotlib.pyplot

class ClusterPipeline:

    def __init__(self, model: str, dim_reduction: str, scaler: str, **kwargs):

        self.model = model
        self.dim_reduction = dim_reduction
        self.scaler = scaler
        self.pipeline: Pipeline = None
        self.added_dict = kwargs

        self._create()

    def init_preprocess(self) -> Pipeline:
        if self.scaler == "Standard":
            return Pipeline([("Standard", StandardScaler())])
        elif self.scaler == "MinMax":
            return Pipeline([("MinMax", MinMaxScaler())])

    def init_dim_reduction(self) -> Pipeline:
        if self.dim_reduction == "PCA":
            return Pipeline([("PCA", PCA(2))])
        if self.dim_reduction == "Tsne":
            return Pipeline([("Tsne", TSNE(2))])

    def init_cluster(self) -> Pipeline:
        if self.model == "Kmeans":
            return Pipeline([("Kmeans", KMeans(**self.added_dict))])

        if self.model == "Spectral":
            return Pipeline([("Spectral", SpectralClustering(**self.added_dict))])

        if self.model == "DBScan":
            return Pipeline([("DBScan", DBSCAN(**self.added_dict))])

        if self.model == "Hierarchical":
            return Pipeline([("Hierarchical", AgglomerativeClustering(**self.added_dict))])

    def _create(self):
        self.pipeline = Pipeline([("preprocessor", self.init_preprocess()),
                                  ("dim reduction", self.init_dim_reduction()),
                                  ("clusterer", self.init_cluster())])

    def fit(self, data: np.array):
        """
        fit the data according to the pipeline parameters
        :param data:
        :return:
        """
        return self.pipeline.fit_transform(data)





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


    fig, ax = plt.subplots()

    for g in np.unique(labels):
        i = np.where(labels == g)
        ax.scatter(Xt[:, 0][i], Xt[:, 1][i], label=g)
    plt.show()



if __name__ == '__main__':
    main()