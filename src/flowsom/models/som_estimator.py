import igraph as ig
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.utils.validation import check_is_fitted

from . import SOM, BaseClusterEstimator, map_data_to_codes
from ._som import eucl, manh

_DISTF_MAP = {
    "euclidean": eucl,
    "manhattan": manh,
}


class SOMEstimator(BaseClusterEstimator):
    """Estimate a Self-Organizing Map (SOM) clustering model."""

    def __init__(
        self,
        xdim=10,
        ydim=10,
        rlen=10,
        mst=1,
        alpha=(0.05, 0.01),
        init=False,
        initf=None,
        distf="euclidean",
        codes=None,
        importance=None,
        seed=None,
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

    def fit(
        self,
        X,
        y=None,
    ):
        """Perform SOM clustering.

        :param inp:  An array of the columns to use for clustering
        :type inp: np.array
        :param xdim: x dimension of SOM
        :type xdim: int
        :param ydim: y dimension of SOM
        :type ydim: int
        :param rlen: Number of times to loop over the training data for each MST
        :type rlen: int
        :param importance: Array with numeric values. Parameters will be scaled
        according to importance
        :type importance: np.array
        """
        codes = self.codes
        xdim = self.xdim
        ydim = self.ydim
        importance = self.importance
        init = self.init
        mst = self.mst
        alpha = self.alpha

        if self.distf not in _DISTF_MAP:
            raise ValueError(f"Unknown distance function '{self.distf}'. Supported: {list(_DISTF_MAP.keys())}")
        distf_func = _DISTF_MAP[self.distf]

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
                raise ValueError(
                    f"codes must have shape ({xdim * ydim}, {X.shape[1]}), got {codes.shape}"
                )

        if importance is not None:
            X = X * np.asarray(importance)[np.newaxis, :]

        # Initialize the grid
        grid = [(x, y) for x in range(xdim) for y in range(ydim)]
        n_codes = len(grid)

        if self.seed is not None:
            np.random.seed(self.seed)

        if codes is None:
            if init:
                codes = self.initf(X, xdim, ydim)
            else:
                codes = X[np.random.choice(X.shape[0], n_codes, replace=False), :]

        # Initialize the neighbourhood
        nhbrdist = squareform(pdist(grid, metric="chebyshev"))

        # Initialize the radius
        radius = (np.quantile(nhbrdist, 0.67), 0)
        if mst == 1:
            radius = [radius]
            alpha = [alpha]
        else:
            radius = np.linspace(radius[0], radius[1], num=mst + 1)
            radius = [tuple(radius[i : i + 2]) for i in range(mst)]
            alpha = np.linspace(alpha[0], alpha[1], num=mst + 1)
            alpha = [tuple(alpha[i : i + 2]) for i in range(mst)]

        # Compute the SOM
        for i in range(mst):
            codes = SOM(
                X,
                codes,
                nhbrdist,
                alphas=alpha[i],
                radii=radius[i],
                ncodes=n_codes,
                rlen=self.rlen,
                distf=distf_func,
                seed=self.seed,
            )
            if mst != 1:
                nhbrdist: list[list[int]] = _dist_mst(codes)

        clusters, dists = map_data_to_codes(data=X, codes=codes, distf=distf_func)
        self.codes, self.labels_, self.distances = codes.copy(), clusters, dists
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
        distf_func = _DISTF_MAP[self.distf]
        clusters, dists = map_data_to_codes(X, self.codes, distf=distf_func)
        self.labels_ = clusters.astype(int)
        self.distances = dists
        return self.labels_

    def fit_predict(self, X, y=None):
        """Fit the model and predict labels."""
        self.fit(X)
        return self.predict(X)


def _dist_mst(codes):
    adjacency = cdist(
        codes,
        codes,
        metric="euclidean",
    )
    full_graph = ig.Graph.Weighted_Adjacency(adjacency, mode="undirected", loops=False)
    MST_graph = ig.Graph.spanning_tree(full_graph, weights=full_graph.es["weight"])
    codes = [
        [len(x) - 1 for x in MST_graph.get_shortest_paths(v=i, to=MST_graph.vs.indices, weights=None)]
        for i in MST_graph.vs.indices
    ]
    return codes
