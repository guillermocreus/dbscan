import numpy as np
from collections import deque
from sklearn.neighbors import KDTree
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array


class DBSCANPP(ClusterMixin, BaseEstimator):

    def __init__(self, m, epsilon=0.5, minPts=10):
        self.m = m
        self.epsilon = epsilon
        self.minPts = minPts

    def k_centers(self, X):
        ind_to_add = 0
        inds_S = [ind_to_add]

        dists_X_to_S = np.full((len(X), 1), np.inf)

        for _ in range(self.m - 1):
            tree = KDTree(X[[ind_to_add]])
            new_dists, _ = tree.query(X, k=1, return_distance=True)
            dists_X_to_S = np.minimum(dists_X_to_S, new_dists)

            ind_to_add = np.argmax(dists_X_to_S)
            inds_S.append(ind_to_add)
        inds_S = np.array(inds_S)

        return inds_S

    def connected_components(self, inds_C, X):
        clusters = np.full(len(X), -1, dtype=np.int32)
        if not len(inds_C):
            return clusters

        tree_C = KDTree(X[inds_C])
        C_neighbors_C = tree_C.query_radius(X[inds_C], r=self.epsilon)
        convert_indices_C = {val: ind for ind, val in enumerate(inds_C)}
        stack = deque()
        current_cluster = 0

        for ind_c in inds_C:
            if clusters[ind_c] == -1:
                stack.append(ind_c)

                while len(stack):
                    node = stack.pop()

                    for neighbor in C_neighbors_C[convert_indices_C[node]]:
                        neighbor = inds_C[neighbor]
                        if clusters[neighbor] == -1:
                            stack.append(neighbor)
                            clusters[neighbor] = current_cluster

                current_cluster += 1

        dist_X_to_C, closest_C_from_X = tree_C.query(X, k=1)
        closest_C_from_X_conv = inds_C[closest_C_from_X.flatten()]

        for i in range(len(X)):
            if clusters[i] == -1:
                clusters[i] = clusters[closest_C_from_X_conv[i]]

        clusters[dist_X_to_C.flatten() > self.epsilon] = -1

        return clusters

    def fit_predict(self, X, sampling='uniform', seed=2):
        X = check_array(X)
        self.X_ = X

        np.random.seed(seed)
        if sampling == 'uniform':
            inds_S = np.random.choice(len(X), self.m, replace=False)
        elif sampling == 'k-center':
            inds_S = self.k_centers(X)
        else:
            inds_S = np.arange(len(X))

        tree_X = KDTree(X)
        dists_S_to_X, _ = tree_X.query(X[inds_S], k=self.minPts)
        # Results are sorted, so we check only the last point
        inds_C = inds_S[dists_S_to_X[:, -1] <= self.epsilon]

        # We build the clusters
        clusters = self.connected_components(inds_C, X)

        # Return the clusters
        return clusters
