import numpy as np
from sklearn.metrics import v_measure_score

from flowsom.models import SOMEstimator


def test_clustering(X):
    som = SOMEstimator()
    y_pred: SOMEstimator = som.fit_predict(X)
    assert y_pred.shape == (100,)


def test_clustering_v_measure(X_and_y):
    som = SOMEstimator(seed=1)
    X, y_true = X_and_y
    y_pred = som.fit_predict(X)
    score = v_measure_score(y_true, y_pred)
    assert score > 0.7


def test_reproducibility_no_seed(X):
    som_1 = SOMEstimator(seed=None)
    som_2 = SOMEstimator(seed=None)
    codes_1 = som_1.fit(X).codes.flatten()
    codes_2 = som_2.fit(X).codes.flatten()

    assert not all(codes_1 == codes_2)


def test_reproducibility_seed(X):
    som_1 = SOMEstimator(seed=1)
    som_2 = SOMEstimator(seed=1)
    codes_1 = som_1.fit(X).codes.flatten()
    codes_2 = som_2.fit(X).codes.flatten()

    assert all(codes_1 == codes_2)


def test_importance_scaling_predict():
    """Test that predict() applies importance scaling consistently with fit()."""
    X = np.random.RandomState(42).rand(200, 4)
    importance = [1.0, 2.0, 0.5, 1.5]
    est = SOMEstimator(xdim=3, ydim=3, importance=importance, seed=42)
    est.fit(X)
    labels_fit = est.labels_.copy()
    labels_predict = est.predict(X)
    np.testing.assert_array_equal(labels_fit, labels_predict)


def test_input_validation():
    """Test that fit() raises ValueError for invalid inputs."""
    import pytest

    est = SOMEstimator(xdim=10, ydim=10)
    # Too few samples for 100 codes
    X_small = np.random.rand(5, 3)
    with pytest.raises(ValueError, match="Number of samples"):
        est.fit(X_small)

    # Empty data
    X_empty = np.empty((0, 3))
    with pytest.raises(ValueError, match="no samples"):
        est.fit(X_empty)
