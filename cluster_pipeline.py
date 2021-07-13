import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Tuple


class TSNEWrapper(TSNE):
    """
    Wrapper class for t-SNE model.
    """
    def __init__(self, components=2):
        """
        Initiate the wrapper using the super constructor.
        :param components: number of dimensions to project on.
        """
        super(TSNEWrapper, self).__init__(n_components=components)

    def transform(self, X: np.array, y: np.array = None):
        """
        Applying the model on a given data.
        :param X: The given data.
        :param y: This parameter should be set always to None.
        :return: The transformed data.
        """
        return self.fit_transform(X)


class ClusterPipeline:
    """
    Class that represents the clustering pipeline.
    """
    def __init__(self, model: str, dim_reduction: Tuple[str, int], scaler: str, **kwargs):
        """
        Initiate the pipeline object.
        :param model: String describing the clustering method to be used.
        Options are: "Kmeans", "MiniBatchKmeans", "Spectral", "DBScan" and "Hierarchical".
        :param dim_reduction: Tuple of (String/None, int) describing the dimension reduction method and the number of dimensions to project on to be used.
        Options are: "PCA", "Tsne"  for the String. The String argument could be None if dimension reduction is unnesseccery.
        :param scaler: String describing the standardization/normalization method to be used.
        Options are: "Standard" and "MinMax".
        :param kwargs: Parameters needed according to the chosen model. For example, n_clusters=k, batch_size=10 if we chose "MiniBatchKmeans" as our model.
        """
        self.model = model
        self.dim_reduction = dim_reduction[0]
        self.number_of_dims = dim_reduction[1]
        self.scaler = scaler
        self.pipeline: Pipeline = None
        self.added_dict = kwargs
        self._create()

    def init_preprocess(self) -> Pipeline:
        """
        Initiate the standardization/normalization layer.
        :return: pipeline object.
        """
        if self.scaler == "Standard":
            return Pipeline([("Standard", StandardScaler())])
        elif self.scaler == "MinMax":
            return Pipeline([("MinMax", MinMaxScaler())])

    def init_dim_reduction(self) -> Pipeline:
        """
        Initiate the dim reduction layer.
        :return: pipeline object.
        """
        if self.dim_reduction == "PCA":
            return Pipeline([("PCA", PCA(self.number_of_dims))])
        if self.dim_reduction == "Tsne":
            return Pipeline([("Tsne", TSNEWrapper(self.number_of_dims))])

    def init_cluster(self) -> Pipeline:
        """
        Initiate the clustering layer.
        :return: pipeline object.
        """
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
        """
        Creates the whole pipeline.
        :return:
        """
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
        Fit the data according to the pipeline parameters
        :param X: The given data.
        :return: The transformed data.
        """
        return self.pipeline.fit(data)