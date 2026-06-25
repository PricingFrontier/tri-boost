from __future__ import annotations

import json
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


def mixed_categorical_fixture() -> tuple[np.ndarray, np.ndarray]:
    n = 96
    num = (np.arange(n) % 8).astype(np.float32)
    levels = np.asarray(["alpha", "beta", "gamma", "delta"], dtype=object)
    brand = levels[np.arange(n) % levels.shape[0]]
    x = np.empty((n, 2), dtype=object)
    x[:, 0] = num
    x[:, 1] = brand
    y = (
        0.15 * num
        + np.where(brand == "beta", 1.25, 0.0)
        + np.where(brand == "gamma", -0.75, 0.25)
    )
    return x, y.astype(np.float32)


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
    out = np.empty_like(pred1)
    returned = est._model.predict(x.astype(np.float32), out=out)
    assert returned is out
    np.testing.assert_array_equal(out, pred1)

    loaded = TriBoostRegressor.from_bytes(est.to_bytes())
    np.testing.assert_array_equal(pred1, loaded.predict(x.astype(np.float32)))
    loaded_json = TriBoostRegressor.from_json(est.to_json())
    np.testing.assert_array_equal(pred1, loaded_json.predict(x.astype(np.float32)))

    export = json.loads(est.tables(x.astype(np.float32), ref_measure="uniform"))
    assert export["mode"] == "Exact"
    assert export["link"] == "Identity"
    assert export["tables"]


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


def test_credibility_floor_params_round_trip_and_path_smooth_shrinks() -> None:
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    # Credibility kwargs survive get_params / clone (the sklearn config contract).
    est = small_regressor(
        min_data_in_leaf=4,
        min_sum_hessian_in_leaf=1.0,
        min_weight_sum_in_leaf=4.0,
        path_smooth=2.0,
    )
    params = est.get_params()
    assert params["min_data_in_leaf"] == 4
    assert params["path_smooth"] == 2.0
    assert clone(est).get_params()["min_weight_sum_in_leaf"] == 4.0

    # path_smooth is value-level: same structure, but it shifts the served predictions,
    # and the model stays exactly decomposable.
    plain = small_regressor().fit(x32, y)
    floored = est.fit(x32, y)
    assert json.loads(floored.tables(x32, ref_measure="uniform"))["mode"] == "Exact"
    assert not np.allclose(plain.predict(x32), floored.predict(x32))


def test_negative_credibility_floor_is_rejected() -> None:
    x, y = regression_fixture()
    with pytest.raises(Exception):
        small_regressor(min_sum_hessian_in_leaf=-1.0).fit(x.astype(np.float32), y)


def test_nesterov_is_rejected_as_unstable() -> None:
    # AGBM/Nesterov acceleration currently diverges (momentum correction unimplemented),
    # so the Python surface refuses it loudly rather than return a blown-up model.
    x, y = regression_fixture()
    with pytest.raises(Exception, match="nesterov"):
        small_regressor(nesterov=True).fit(x.astype(np.float32), y)


def test_all_categorical_input_stays_exact_and_predicts() -> None:
    # No numeric features at all — the model is built entirely from native categoricals.
    rng = np.random.default_rng(0)
    n = 1500
    a = rng.integers(0, 5, n)
    b = rng.integers(0, 4, n)
    x = np.array([[f"A{ai}", f"B{bi}"] for ai, bi in zip(a, b)], dtype=object)
    y = (a + 2.0 * b).astype(np.float32)
    m = TriBoostRegressor(
        objective="squared_error", n_trees=80, learning_rate=0.1, seed=0,
        categorical_features=[0, 1],
    ).fit(x, y)
    pred = np.asarray(m.predict(x))
    assert pred.shape == (n,) and np.isfinite(pred).all()
    exp = json.loads(m.tables(x[:128]))
    assert exp["mode"] == "Exact"


def test_empty_feature_matrix_is_rejected() -> None:
    with pytest.raises(Exception, match="at least one feature"):
        TriBoostRegressor(n_trees=5, seed=0).fit(
            np.empty((50, 0), dtype=np.float32), np.zeros(50, dtype=np.float32)
        )


def test_booster_knobs_round_trip_and_stay_exact() -> None:
    # The §06/§09/§07 levers (ensemble, sampling, hist precision, refit, interaction order)
    # are now reachable from Python; they survive get_params/clone and stay exactly
    # decomposable (every booster is leaf-scalar / tree-alpha / intercept level).
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    est = small_regressor(
        n_bags=3,
        subsample=0.8,
        hist_precision="quantized",
        ridge_refit_l2=0.5,
        random_strength=0.1,
        reanchor=True,
        max_interaction_order=2,
    )
    params = est.get_params()
    assert params["n_bags"] == 3
    assert params["hist_precision"] == "quantized"
    assert params["max_interaction_order"] == 2
    assert clone(est).get_params()["subsample"] == 0.8

    est.fit(x32, y)
    assert json.loads(est.tables(x32, ref_measure="uniform"))["mode"] == "Exact"
    assert est.predict(x32).shape == (x32.shape[0],)


def test_new_accuracy_knobs_round_trip_and_stay_exact() -> None:
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    est = small_regressor(
        l1_leaf=0.01,
        colsample_bytree=0.67,
        learning_rate_decay=0.05,
        validation_fraction=0.2,
        early_stopping_rounds=3,
        leaf_refine_steps=1,
        leaf_refine_backtracks=3,
    )
    params = est.get_params()
    assert params["l1_leaf"] == 0.01
    assert params["colsample_bytree"] == 0.67
    assert params["learning_rate_decay"] == 0.05
    assert params["validation_fraction"] == 0.2
    assert params["early_stopping_rounds"] == 3
    assert params["leaf_refine_steps"] == 1
    assert params["leaf_refine_backtracks"] == 3
    assert clone(est).get_params()["colsample_bytree"] == 0.67

    est.fit(x32, y)
    assert json.loads(est.tables(x32, ref_measure="uniform"))["mode"] == "Exact"
    assert est.predict(x32).shape == (x32.shape[0],)


def test_new_accuracy_invalid_params_are_rejected() -> None:
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    for kwargs in (
        {"l1_leaf": -1.0},
        {"colsample_bytree": 0.0},
        {"learning_rate_decay": -0.1},
        {"validation_fraction": 0.0},
        {"validation_fraction": 0.2, "early_stopping_rounds": 0},
        {"leaf_refine_steps": 1, "leaf_refine_backtracks": 0},
    ):
        with pytest.raises(Exception):
            small_regressor(**kwargs).fit(x32, y)


def test_regressor_native_categorical_object_array_stays_exact_and_cloneable() -> None:
    x, y = mixed_categorical_fixture()
    est = small_regressor(
        categorical_features=[1],
        cat_smooth=5.0,
        cat_target="mean",
        cat_leakage="kfold",
        cat_k=3,
        cat_min_data_per_group=0.0,
    )
    params = est.get_params()
    assert params["categorical_features"] == [1]
    assert params["cat_smooth"] == 5.0
    assert params["cat_target"] == "mean"
    assert params["cat_leakage"] == "kfold"
    assert params["cat_k"] == 3
    assert params["cat_min_data_per_group"] == 0.0
    assert clone(est).get_params()["categorical_features"] == [1]

    with pytest.warns(PrecisionWarning):
        est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)
    assert est.n_features_in_ == 2
    assert est._cat_indices_ == [1]
    export = json.loads(est.tables(x, ref_measure="uniform"))
    assert export["mode"] == "Exact"

    loaded = TriBoostRegressor.from_bytes(est.to_bytes())
    loaded.categorical_features = [1]
    with pytest.warns(PrecisionWarning):
        loaded_pred = loaded.predict(x)
    np.testing.assert_array_equal(loaded_pred, pred)


def test_native_categorical_early_stopping_kfold_allowed_ordered_gated() -> None:
    # KFold cross-fit (the default) gives OOF per-row encodings, so the carved validation
    # fold excludes each row's own target → internal early-stopping is leakage-free + allowed.
    x, y = mixed_categorical_fixture()
    est = small_regressor(
        categorical_features=[1], validation_fraction=0.2, cat_leakage="kfold"
    )
    est.fit(x, y)  # must NOT raise
    assert np.isfinite(np.asarray(est.predict(x))).all()
    # ordered/loo keep the guard (subtler leakage profiles with internal early stopping).
    with pytest.raises(Exception, match="validation_fraction"):
        small_regressor(
            categorical_features=[1], validation_fraction=0.2, cat_leakage="ordered"
        ).fit(x, y)


def test_regressor_native_categorical_dataframe_names_and_monotone_guard() -> None:
    pd = pytest.importorskip("pandas")
    x, y = mixed_categorical_fixture()
    frame = pd.DataFrame(
        {
            "brand": x[:, 1],
            "age": x[:, 0].astype(np.float32),
        }
    )

    est = small_regressor(categorical_features=["brand"])
    est.fit(frame, y)
    assert est.feature_names_in_.tolist() == ["brand", "age"]
    assert est._model.feature_names == ["age", "brand"]
    pred = est.predict(frame)
    assert pred.shape == (frame.shape[0],)

    with pytest.raises(ValueError, match="monotone_constraints"):
        small_regressor(
            categorical_features=["brand"],
            monotone_constraints={"age": 1},
        ).fit(frame, y)


def test_classifier_native_categorical_predict_proba() -> None:
    x, y_reg = mixed_categorical_fixture()
    y = np.where(y_reg > np.median(y_reg), "high", "low")
    clf = TriBoostClassifier(
        n_trees=18,
        learning_rate=0.2,
        lambda_=1.0,
        max_bin=32,
        seed=11,
        categorical_features=[1],
    )
    with pytest.warns(PrecisionWarning):
        clf.fit(x, y)
    proba = clf.predict_proba(x)
    assert proba.shape == (x.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6)
    export = json.loads(clf.tables(x, ref_measure="uniform"))
    assert export["mode"] == "Exact"


def test_outer_bag_is_thread_count_deterministic() -> None:
    # Bagging folds convex weights into tree alphas — still byte-identical across n_jobs.
    x, y = regression_fixture()
    a = small_regressor(n_bags=4, n_jobs=1).fit(x.astype(np.float32), y).to_bytes()
    b = small_regressor(n_bags=4, n_jobs=2).fit(x.astype(np.float32), y).to_bytes()
    assert a == b


def test_invalid_hist_precision_is_rejected() -> None:
    x, y = regression_fixture()
    with pytest.raises(Exception):
        small_regressor(hist_precision="nonsense").fit(x.astype(np.float32), y)


def test_reanchor_defaults_on_for_log_link_only() -> None:
    # reanchor=None ⇒ link-aware default: ON for log-link (gamma/poisson/tweedie),
    # OFF for identity/logit. Removes post-shrinkage aggregate bias for free.
    rng = np.random.RandomState(0)
    x = rng.rand(400, 3).astype(np.float32)
    y = (1.0 + 2.0 * x[:, 0] + 0.5 * rng.rand(400)).astype(np.float32)  # positive (gamma-safe)
    common = dict(n_trees=40, max_bin=32, seed=0)
    # Gamma (log link): default reanchors → differs from explicitly-off.
    g_def = TriBoostRegressor(objective="gamma", **common).fit(x, y).predict(x)
    g_off = TriBoostRegressor(objective="gamma", reanchor=False, **common).fit(x, y).predict(x)
    assert not np.allclose(g_def, g_off)
    # SquaredError (identity link): default does NOT reanchor → identical to explicitly-off.
    s_def = TriBoostRegressor(objective="squared_error", **common).fit(x, y).predict(x)
    s_off = TriBoostRegressor(
        objective="squared_error", reanchor=False, **common
    ).fit(x, y).predict(x)
    np.testing.assert_array_equal(s_def, s_off)
