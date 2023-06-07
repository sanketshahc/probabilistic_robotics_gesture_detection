import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# def quantize(data,M = 3):
#     """
#     intakes pd datafram or numpy array
#     SHUOLD BE QUANTIZED OUTSIDE MODEL, because need to use same space for ALL sequences,
#     of ALL classes.
#     :return: series
#     """
#     # assert type(data) == pd.DataFrame
#     assert data.ndim == 2
#     KM = KMeans(M,max_iter=100)
#     ret = pd.Series(KM.fit_predict(data))
#     # ret = ret.rename(dir.split('/')[-1][:-4])
#     return ret


class Kmeans(object):
    def __init__(self, k, seed=None, max_iter=100):
        self.k = k
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter

    def initialize(self, data):
        centroids = np.random.permutation(data.shape[0])[:self.k]
        self.centroids = data[centroids]
        return self.centroids

    def assign_clusters(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        dist = pairwise_distances(data, self.centroids, metric='euclidean')
        self.clusters = np.argmin(dist, axis=1)

        return self.clusters

    def update(self, data):
        self.centroids = np.array(
            [data[self.clusters == i].mean(axis=0) for i in range(self.k)])
        return self.centroids

    def predict(self, data):
        return self.assign_clusters(data)

    def fit(self, data):
        self.centroids = self.initialize(data)
        for iter in range(self.max_iter):
            self.clusters = self.assign_clusters(data)
            self.centroids = self.update(data)
        return self

# X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
#
# kmeans = Kmeans(3,max_iter=100)
# y1 = kmeans.fit(X).predict(X)
# y2 = quantize(X)
#
# plt.scatter(X[:,0], X[:,1], c=y1)
# plt.show()
# plt.scatter(X[:,0], X[:,1], c=y2)
# plt.show()
