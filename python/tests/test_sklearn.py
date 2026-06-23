from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tri_boost.sklearn import PrecisionWarning, TriBoostClassifier, TriBoostRegressor


def regression_fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[float(i % 6), float((i // 2) % 5), float((i // 3) % 4)] for i in range(96)],
        dtype=np.float64,
    )
    y = np.where(x[:, 0] <= 2.0, 1.5, -0.75) + np.where(x[:, 1] <= 2.0, 0.5, -0.25)
    return x, y.astype(np.float32)


def classifier_fixture() -> tuple[np.ndarray, np.ndarray]:
    x, y_reg = regression_fixture()
    y = np.where(y_reg > np.median(y_reg), "yes", "no")
    return x, y


def small_regressor(**kwargs) -> TriBoostRegressor:
    return TriBoostRegressor(
        n_trees=16,
        learning_rate=0.25,
        lambda_=1.0,
        max_bin=32,
        seed=7,
        **kwargs,
    )


def test_regressor_fit_predict_serialize_and_warns_once() -> None:
    x, y = regression_fixture()
    est = small_regressor()
    with pytest.warns(PrecisionWarning):
        est.fit(x, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred1 = est.predict(x)
        pred2 = est.predict(x)
    assert not [w for w in caught if issubclass(w.category, PrecisionWarning)]
    np.testing.assert_array_equal(pred1, pred2)

    loaded = TriBoostRegressor.from_bytes(est.to_bytes())
    np.testing.assert_array_equal(pred1, loaded.predict(x.astype(np.float32)))
    loaded_json = TriBoostRegressor.from_json(est.to_json())
    np.testing.assert_array_equal(pred1, loaded_json.predict(x.astype(np.float32)))


def test_regressor_clone_set_params_and_not_fitted_contract() -> None:
    x, y = regression_fixture()
    est = small_regressor()
    cloned = clone(est)
    assert cloned.get_params()["n_trees"] == 16
    with pytest.raises(NotFittedError):
        cloned.predict(x)

    est.fit(x.astype(np.float32), y)
    est.set_params(n_trees=4)
    with pytest.raises(NotFittedError):
        est.predict(x.astype(np.float32))


def test_classifier_predict_proba_classes_and_roundtrip() -> None:
    x, y = classifier_fixture()
    clf = TriBoostClassifier(
        n_trees=18,
        learning_rate=0.2,
        lambda_=1.0,
        max_bin=32,
        seed=11,
    )
    with pytest.warns(PrecisionWarning):
        clf.fit(x, y)
    assert clf.classes_.tolist() == ["no", "yes"]
    proba = clf.predict_proba(x.astype(np.float32))
    assert proba.shape == (x.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6)
    pred = clf.predict(x.astype(np.float32))
    assert set(pred.tolist()) <= {"no", "yes"}

    loaded = TriBoostClassifier.from_bytes(clf.to_bytes())
    assert loaded.classes_.tolist() == ["no", "yes"]
    np.testing.assert_array_equal(proba, loaded.predict_proba(x.astype(np.float32)))


def test_python_fit_is_thread_count_deterministic() -> None:
    x, y = regression_fixture()
    a = small_regressor(n_jobs=1).fit(x.astype(np.float32), y).to_bytes()
    b = small_regressor(n_jobs=2).fit(x.astype(np.float32), y).to_bytes()
    assert a == b
