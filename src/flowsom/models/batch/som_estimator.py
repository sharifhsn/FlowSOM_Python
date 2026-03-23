import numpy as np

from flowsom.models._base_som_estimator import _BaseSOMEstimator
from flowsom.models._som import manh

from . import SOM_Batch, map_data_to_codes
from ._som import eucl_without_sqrt

_DISTF_MAP_BATCH = {
    "euclidean": eucl_without_sqrt,
    "manhattan": manh,
}


class BatchSOMEstimator(_BaseSOMEstimator):
    """Estimate a Self-Organizing Map (SOM) clustering model using batch training."""

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
        num_batches=10,
        seed=None,
    ):
        super().__init__(
            xdim=xdim, ydim=ydim, rlen=rlen, mst=mst, alpha=alpha,
            init=init, initf=initf, distf=distf, codes=codes,
            importance=importance, seed=seed,
        )
        self.num_batches = num_batches

    def _get_distf_map(self):
        return _DISTF_MAP_BATCH

    def _train_som(self, X, codes, nhbrdist, alpha, radius, n_codes, distf_func):
        num_batches = self.num_batches

        # Split data into interleaved batches
        data = [X[i::num_batches, :] for i in range(num_batches)]

        # Equalize batch sizes by duplicating last row
        for i in range(num_batches):
            if data[i].shape[0] < data[0].shape[0]:
                data[i] = np.vstack([data[i], X[-1, :]])

        return SOM_Batch(
            np.array(data, dtype=np.float32),
            codes, nhbrdist,
            alphas=alpha, radii=radius, ncodes=n_codes,
            rlen=self.rlen, num_batches=num_batches,
            distf=distf_func, seed=self.seed,
        )

    def _map_data_to_codes(self, X, codes, distf_func):
        return map_data_to_codes(data=X, codes=codes, metric=self.distf)
