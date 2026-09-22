import numpy as np
import pytest

from laya_vision_stitch.alignment import RidgeConnector, agreement


def test_connector_recovers_unseen_linear_mapping_and_roundtrips(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 6))
    w = rng.normal(size=(6, 4))
    y = x @ w + 2
    model = RidgeConnector().fit(x[:60], y[:60], 1e-6)
    assert np.max(np.abs(model.predict(x[60:]) - y[60:])) < 1e-4
    path = tmp_path / "connector.npz"
    model.save(path)
    np.testing.assert_allclose(RidgeConnector.load(path).predict(x[60:]), model.predict(x[60:]))


def test_scaling_uses_only_fitting_data_and_handles_constant_features():
    x = np.array([[1, 2], [3, 2], [5, 2]])
    model = RidgeConnector().fit(x, np.array([[0], [1], [2]]), 1)
    before = model.mean.copy()
    assert np.isfinite(model.predict([[1e6, 2]])).all()
    np.testing.assert_array_equal(model.mean, before)
    np.testing.assert_array_equal(model.mean, [3, 2])


def test_nonfinite_features_and_nonpositive_regularization_are_rejected():
    with pytest.raises(ValueError):
        RidgeConnector().fit([[0], [1]], [[0], [1]], 0)
    with pytest.raises(ValueError):
        RidgeConnector().fit([[0], [np.nan]], [[0], [1]], 1)


def test_balanced_agreement_exposes_majority_prediction():
    result = agreement(["A"] * 9 + ["G"], ["A"] * 10)
    assert result["agreement"] == 0.9
    assert result["balanced_agreement"] == 0.5
