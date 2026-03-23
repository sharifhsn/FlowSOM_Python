from . import SOM, map_data_to_codes
from ._base_som_estimator import _BaseSOMEstimator
from ._som import eucl, manh

_DISTF_MAP = {
    "euclidean": eucl,
    "manhattan": manh,
}


class SOMEstimator(_BaseSOMEstimator):
    """Estimate a Self-Organizing Map (SOM) clustering model."""

    def _get_distf_map(self):
        return _DISTF_MAP

    def _train_som(self, X, codes, nhbrdist, alpha, radius, n_codes, distf_func):
        return SOM(
            X, codes, nhbrdist,
            alphas=alpha, radii=radius, ncodes=n_codes,
            rlen=self.rlen, distf=distf_func, seed=self.seed,
        )

    def _map_data_to_codes(self, X, codes, distf_func):
        return map_data_to_codes(data=X, codes=codes, distf=distf_func)
