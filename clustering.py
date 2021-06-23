import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from  sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np


class clusterPipline:

    clustering_set = set("Kmeans", "Spectral", "DBScan", "Hierarchical")

    def __init__(self, model:str, dim_reduction: bool, scaler: str, **kwargs):

        self.model = model
        self.dim_reduction = dim_reduction
        self.scaler = scaler
        self.preprocessor_pipeline = None
        self.dim_reduction_pipeline = None
        self.cluster_pipeline = None
        self.__dict__ = kwargs


    def init_preprocess(self ):
        if self.scaler == "Standard":
            self.preprocessor_pipeline = Pipeline([("scaler", StandardScaler())])
        elif self.scaler == "MinMax":
            self.preprocessor_pipeline = Pipeline([("scaler", MinMaxScaler())])

    def init_dim_reduction(self):
        if self.dim_reduction == "PCA":
            self.dim_reduction_pipeline = Pipeline(["dim reduce", PCA(2)])
        if self.dim_reduction == "Tsne":
            self.dim_reduction_pipeline = Pipeline(["dim reduce", TSNE(2)])

    def init_cluster(self):
        if self.model == "Kmeans"


    def fit(self, data: np.array):
        """
        fit the data according to the pipeline parameters
        :param data:
        :return:
        """
        return


class KMeans: