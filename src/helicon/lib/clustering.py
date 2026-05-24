import numpy as np
from sklearn.cluster import AgglomerativeClustering

__all__ = [
    "AgglomerativeClusteringWithMinSize",
]


class AgglomerativeClusteringWithMinSize(AgglomerativeClustering):
    """Agglomerative clustering that ensures a minimum cluster size.

    Extends scikit-learn's AgglomerativeClustering by iteratively
    reassigning points from clusters smaller than *min_cluster_size*
    to the nearest valid cluster.

    Parameters
    ----------
    min_cluster_size : int, optional
        Minimum number of points per cluster. Defaults to 2.
    n_clusters : int, optional
        Number of clusters. Defaults to 2.
    metric : str, optional
        Distance metric. Defaults to ``"euclidean"``.
    memory : object, optional
        Caching object. Defaults to None.
    connectivity : array-like, optional
        Connectivity matrix. Defaults to None.
    compute_full_tree : str or bool, optional
        Whether to compute the full tree. Defaults to ``"auto"``.
    linkage : str, optional
        Linkage criterion. Defaults to ``"ward"``.
    distance_threshold : float, optional
        Distance threshold. Defaults to None.
    """

    def __init__(
        self,
        min_cluster_size=2,
        n_clusters=2,
        metric="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        distance_threshold=None,
    ):
        super().__init__(
            n_clusters=n_clusters,
            metric=metric,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
        )
        self.min_cluster_size = min_cluster_size

    def fit(self, X, y=None):
        """Fit the clustering model and enforce minimum cluster size.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API compatibility.

        Returns
        -------
        self
            Fitted estimator.
        """
        super().fit(X, y)
        labels = self.labels_

        while True:
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) < 3:
                break

            small_clusters = unique[counts < self.min_cluster_size]
            if len(small_clusters) == 0:
                break

            # If all clusters are small, merge the two smallest
            if len(small_clusters) == len(unique):
                smallest_two = unique[np.argsort(counts)[:2]]
                labels[labels == smallest_two[1]] = smallest_two[0]
                continue

            from sklearn.metrics import pairwise_distances

            distances = pairwise_distances(X)
            for small_cluster in small_clusters:
                small_cluster_points = np.where(labels == small_cluster)[0]
                for point in small_cluster_points:
                    # Find the nearest point not in a small cluster
                    valid_points = np.where(~np.isin(labels, small_clusters))[0]
                    nearest_point = valid_points[
                        np.argmin(distances[point, valid_points])
                    ]
                    labels[point] = labels[nearest_point]

        self.labels_ = labels
        self.n_clusters_ = len(np.unique(labels))

        return self
