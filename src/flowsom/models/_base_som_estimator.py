"""Shared base class for SOM estimators."""

from __future__ import annotations

import igraph as ig
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.utils.validation import check_is_fitted

from .base_cluster_estimator import BaseClusterEstimator


class _BaseSOMEstimator(BaseClusterEstimator):
    """Base class with shared grid setup, validation, and scheduling logic.

    Subclasses must implement ``_train_som`` and ``_map_data_to_codes``.
    """

    def __init__(
        self,
        xdim: int = 10,
        ydim: int = 10,
        rlen: int = 10,
        mst: int = 1,
        alpha: tuple[float, float] = (0.05, 0.01),
        init: bool = False,
        initf=None,
        distf: str = "euclidean",
        codes: np.ndarray | None = None,
        importance: np.ndarray | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.rlen = rlen
        self.mst = mst
        self.alpha = alpha
        self.init = init
        self.initf = initf
        self.distf = distf
        self.codes = codes
        self.importance = importance
        self.seed = seed

    # -- hooks for subclasses ------------------------------------------------

    def _get_distf_map(self) -> dict:
        """Return mapping from distance name to callable."""
        raise NotImplementedError

    def _train_som(self, X, codes, nhbrdist, alpha, radius, n_codes, distf_func):
        """Run one round of SOM training. Must return updated codes."""
        raise NotImplementedError

    def _map_data_to_codes(self, X, codes, distf_func):
        """Map data points to nearest codes. Must return (labels, distances)."""
        raise NotImplementedError

    # -- shared implementation -----------------------------------------------

    def fit(self, X, y=None):
        """Perform SOM clustering."""
        codes = self.codes
        xdim = self.xdim
        ydim = self.ydim
        importance = self.importance
        init = self.init
        mst = self.mst
        alpha = self.alpha

        distf_map = self._get_distf_map()
        if self.distf not in distf_map:
            raise ValueError(f"Unknown distance function '{self.distf}'. Supported: {list(distf_map.keys())}")
        distf_func = distf_map[self.distf]

        n_codes_expected = xdim * ydim
        if X.shape[0] == 0:
            raise ValueError("Input data X has no samples")
        if X.shape[0] < n_codes_expected:
            raise ValueError(
                f"Number of samples ({X.shape[0]}) must be >= number of codes "
                f"({n_codes_expected} = {xdim}x{ydim})"
            )

        if codes is not None:
            if codes.shape[1] != X.shape[1] or codes.shape[0] != xdim * ydim:
                raise ValueError(f"codes must have shape ({xdim * ydim}, {X.shape[1]}), got {codes.shape}")

        if importance is not None:
            X = X * np.asarray(importance)[np.newaxis, :]

        grid = [(x, y) for x in range(xdim) for y in range(ydim)]
        n_codes = len(grid)

        if self.seed is not None:
            np.random.seed(self.seed)

        if codes is None:
            if init:
                codes = self.initf(X, xdim, ydim)
            else:
                codes = X[np.random.choice(X.shape[0], n_codes, replace=False), :]

        nhbrdist = squareform(pdist(grid, metric="chebyshev"))

        radius = (np.quantile(nhbrdist, 0.67), 0)
        if mst == 1:
            radius = [radius]
            alpha = [alpha]
        else:
            radius = np.linspace(radius[0], radius[1], num=mst + 1)
            radius = [tuple(radius[i : i + 2]) for i in range(mst)]
            alpha = np.linspace(alpha[0], alpha[1], num=mst + 1)
            alpha = [tuple(alpha[i : i + 2]) for i in range(mst)]

        for i in range(mst):
            codes = self._train_som(X, codes, nhbrdist, alpha[i], radius[i], n_codes, distf_func)
            if mst != 1:
                nhbrdist = _dist_mst(codes)

        clusters, dists = self._map_data_to_codes(X, codes, distf_func)
        self.codes, self.labels_, self.distances = codes.copy(), clusters.astype(int), dists
        self._is_fitted = True
        return self

    def predict(self, X, y=None):
        """Predict cluster labels for new data.

        Note: Updates self.labels_ and self.distances as a side effect.
        This is used internally by FlowSOM.new_data().
        """
        check_is_fitted(self)
        if self.importance is not None:
            X = X * np.asarray(self.importance)[np.newaxis, :]
        distf_func = self._get_distf_map()[self.distf]
        clusters, dists = self._map_data_to_codes(X, self.codes, distf_func)
        self.labels_ = clusters.astype(int)
        self.distances = dists
        return self.labels_

    def fit_predict(self, X, y=None):
        """Fit the model and predict labels."""
        self.fit(X)
        return self.labels_


def _dist_mst(codes):
    """Compute MST-based neighborhood distances from code vectors."""
    adjacency = cdist(codes, codes, metric="euclidean")
    full_graph = ig.Graph.Weighted_Adjacency(adjacency, mode="undirected", loops=False)
    MST_graph = ig.Graph.spanning_tree(full_graph, weights=full_graph.es["weight"])
    return [
        [len(x) - 1 for x in MST_graph.get_shortest_paths(v=i, to=MST_graph.vs.indices, weights=None)]
        for i in MST_graph.vs.indices
    ]
