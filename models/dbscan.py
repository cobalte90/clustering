import numpy as np
from tqdm import tqdm

class DBSCAN:
    def __init__(self, eps: float=0.1, min_samples: int=5, verbose=True):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.verbose = verbose
    
    def _euclid_dist(self, p1, p2) -> float:
        p1, p2 = np.array(p1), np.array(p2)
        return np.linalg.norm(p1 - p2)

    def _find_neighbours(self, X, i) -> np.array:
        neighbours = []
        for potential_neighbour in range(len(X)):
            if self._euclid_dist(X[i], X[potential_neighbour]) <= self.eps:
                neighbours.append(potential_neighbour)
        return np.array(neighbours) # neighbours of current point
    
    def _expand_cluster(self, X, i, neighbours, n_cluster):
        # recursively expand cluster from core point
        for neighbour in neighbours:
            # if point is unvisited (-1), add to current cluster
            if self.labels[neighbour] == -1:
                self.labels[neighbour] = n_cluster
                new_neighbours = self._find_neighbours(X, neighbour)
                if len(new_neighbours) >= self.min_samples:
                    self._expand_cluster(X, neighbour, new_neighbours, n_cluster)
            # if point was marked as noise ( == 0), add to current cluster
            elif self.labels[neighbour] == 0:
                self.labels[neighbour] = n_cluster
                
    def fit(self, X):
        X = np.array(X)
        self.labels = np.full(len(X), -1)
        n_cluster = 0 # current cluster id
        for i in tqdm(range(len(X)), desc="Clustering in progress"):
            # skip visited points
            if self.labels[i] != -1:
                continue
            neighbours = self._find_neighbours(X, i)
            if len(neighbours) < self.min_samples:
                self.labels[i] = 0
                continue
            # start new cluster
            n_cluster += 1
            self.labels[i] = n_cluster
            self._expand_cluster(X, i, neighbours, n_cluster)
        return self.labels
