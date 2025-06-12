import numpy as np
from tqdm import tqdm

class AgglomerativeClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters # int
        self.labels = None # 1-d np.array
        self.pairwise_dist_matrix = None # 2-d np.array
        self.current_n_clusters = None
        self.clusters = None
        self.dendrogram = []
    
    def _euclidian_dist(self, p1, p2) -> float:
        return np.linalg.norm(p1 - p2)
    
    def _compute_pairwise_dist(self, X): # pairwise distances matrix (square)
        self.pairwise_dist_matrix = np.zeros((len(X), len(X)), dtype=float)
        for i in range(len(X)):
            for j in range(len(X)):
                self.pairwise_dist_matrix[i, j] = self._euclidian_dist(X[i], X[j])
    
    def _compute_linkage(self, cluster_1, cluster_2): # average distance between two clusters
        dist_sum = 0
        for i in cluster_1:
            for j in cluster_2:
                dist_sum += self.pairwise_dist_matrix[i, j]
        return dist_sum / (len(cluster_1) * len(cluster_2))

    def fit(self, X):
        X = np.array(X)
        self.clusters = [[i] for i in range(len(X))]
        self.current_n_clusters = len(self.clusters)
        self._compute_pairwise_dist(X)
        with tqdm(total=len(X)-self.n_clusters, desc="Cluster merging") as pbar:
            while self.current_n_clusters > self.n_clusters:
                min_linkage = np.inf
                the_nearest_clusters_idxs = ()
                for i in range(self.current_n_clusters):
                    for j in range(i + 1, self.current_n_clusters):
                        current_linkage = self._compute_linkage(self.clusters[i], self.clusters[j])
                        if current_linkage < min_linkage:
                            min_linkage = current_linkage
                            the_nearest_clusters_idxs = (i, j)

                # add new cluster (cluster_1 + cluster_2)
                new_cluster = self.clusters[the_nearest_clusters_idxs[0]] + self.clusters[the_nearest_clusters_idxs[1]]
                self.clusters.append(new_cluster)
                # drop old clusters
                del self.clusters[max(the_nearest_clusters_idxs)]
                del self.clusters[min(the_nearest_clusters_idxs)]

                # add a cluster merge note to the dendrogram
                self.dendrogram.append({
                    'merged_clusters_idxs': (the_nearest_clusters_idxs[0], the_nearest_clusters_idxs[1]),
                    'new_cluster': new_cluster,
                    'linkage': min_linkage,
                    'remaining_clusters_n': self.current_n_clusters - 1
                })
                self.current_n_clusters -= 1
                pbar.update(1)

        self.labels = np.zeros(len(X), dtype=float)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                self.labels[sample_idx] = cluster_idx
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels