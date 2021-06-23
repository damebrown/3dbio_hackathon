import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np


class ClusterPipeline:
    clustering_set = set("Kmeans", "Spectral", "DBScan", "Hierarchical")

    def __init__(self, model: str, dim_reduction: bool, scaler: str, **kwargs):

        self.model = model
        self.dim_reduction = dim_reduction
        self.scaler = scaler
        self.__dict__ = kwargs
        self.pipeline = None
        self._create()

    def init_preprocess(self) -> Pipeline:
        if self.scaler == "Standard":
            return Pipeline([("Standard", StandardScaler())])
        elif self.scaler == "MinMax":
            return Pipeline([("MinMax", MinMaxScaler())])

    def init_dim_reduction(self) -> Pipeline:
        if self.dim_reduction == "PCA":
            return Pipeline(["PCA", PCA(2)])
        if self.dim_reduction == "Tsne":
            return Pipeline(["Tsne", TSNE(2)])

    def init_cluster(self) -> Pipeline:
        if self.model == "Kmeans":
            return Pipeline([("Kmeans", KMeans(**self.__dict__))])

        if self.model == "Spectral":
            return Pipeline([("Spectral", SpectralClustering(**self.__dict__))])

        if self.model == "DBScan":
            return Pipeline([("DBScan", DBSCAN(**self.__dict__))])

        if self.model == "Hierarchical":
            return Pipeline([("Hierarchical", AgglomerativeClustering(**self.__dict__))])

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
        self.pipeline.fit(data)
