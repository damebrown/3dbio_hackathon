import matplotlib.pyplot as plt
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np


import matplotlib.pyplot

class TSNE_wrapper(TSNE):

    def transform(self, X, y=None):
        return TSNE().fit_transform(X)


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
            return Pipeline([("Tsne", TSNE_wrapper(2))])

    def init_cluster(self) -> Pipeline:
        if self.model == "Kmeans":
            return Pipeline([("Kmeans", KMeans(**self.added_dict))])

        if self.model == "MiniBatchKmeans":
            return Pipeline([("MiniBatchKmeans", MiniBatchKMeans(**self.added_dict))])

        if self.model == "Spectral":
            return Pipeline([("Spectral", SpectralClustering(**self.added_dict))])

        if self.model == "DBScan":
            return Pipeline([("DBScan", DBSCAN(**self.added_dict))])

        if self.model == "Hierarchical":
            return Pipeline([("Hierarchical", AgglomerativeClustering(**self.added_dict))])

    def _create(self):
        pipes = [pipe for pipe in [("preprocessor", self.init_preprocess()), ("dim reduction", self.init_dim_reduction()), ("clusterer", self.init_cluster())] if pipe[1]]
        self.pipeline = Pipeline(pipes)

    def fit_transform(self, data: np.array):
        """
        fit the data according to the pipeline parameters
        :param data:
        :return:
        """
        return self.pipeline.fit_transform(data)


    def fit(self, data: np.array):
        """
        fit the data according to the pipeline parameters
        :param data:
        :return:
        """
        return self.pipeline.fit(data)


# class IterativeCluster:
#
#     def __init__(self, worker_num: int, batch_size: int, data_dir: str):
#
#         self.centroids = []
#         self.worker_num = worker_num
#         self.batch_size = batch_size
#         self.data_dir = data_dir
#
#
#     def cluster(self):
