import numpy as np
from random import randint
from tqdm import tqdm

class my_kmeans:
    def __init__(self, k_clusters=3, max_iter=300, tol=0.0001, verbose=True):
        self.k_clusters = k_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.centroids = None
        self.labels = None
    
    def _euclid_dist(self, x1y1, x2y2) -> float: # returns euclid distance between two objects
        quad_dist = 0
        for i in range(len(x1y1)):
            quad_dist += (x1y1[i] - x2y2[i])**2
        return np.sqrt(quad_dist)
    
    def _find_nearest_centroid(self, object) -> int: # returns cluster number
        min_dist = 9999999999
        cluster_label = None
        for centroid in range(len(self.centroids)):
            dist = self._euclid_dist(object, self.centroids[centroid])
            if dist < min_dist:
                min_dist = dist
                cluster_label = centroid
        return cluster_label
    
    def fit(self, X):
        progress = tqdm(range(self.max_iter), desc="Clustering in progress", disable=not self.verbose) # for progress bar
        X = np.array(X)
        # random clusters initialization
        self.centroids = X[np.random.choice(len(X), self.k_clusters, replace=False)]
        self.labels = np.zeros(len(X), dtype=float)
        for _ in progress:
            # find the nearest centroids for each object
            for x_i in range(len(X)):
                self.labels[x_i] = self._find_nearest_centroid(X[x_i])
            # now self.labels includes cluster number for each object
            for k in range(self.k_clusters):
                current_cluster_objects = np.array(X[self.labels == k])
                new_cluster = current_cluster_objects.mean(axis=0)
                self.centroids[k] = new_cluster
        return self.labels
    
    def predict(self, X):
        X = np.array(X)
        pred_labels = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            pred_labels[i] = self._find_nearest_centroid(X[i])
        return pred_labels