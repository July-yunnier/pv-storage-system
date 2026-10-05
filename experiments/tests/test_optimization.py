import numpy as np
import pandas as pd
import pytest
import h5py

from experiments import optimize
from experiments.optimize import blend_weights, build_features, sky_descriptors
from experiments.predict_optimized import predict_feature_matrices
from experiments.protocol import sample_indices


def fixture_data():
    ts = pd.date_range("2020-01-01", periods=96 * 12, freq="15min", tz="Asia/Shanghai")
    power = np.maximum(0, np.sin((ts.hour.to_numpy() - 6) / 12 * np.pi)) * 100
    frame = pd.DataFrame(dict(ts=ts, power_kw=power, tsi=power * 8, dni=power * 5,
                              ghi=power * 6, temperature=np.full(len(ts), 20.)))
    split = dict(train_end=ts[96 * 7].isoformat(), val_end=ts[96 * 9].isoformat(),
                 test_end=(ts[-1] + pd.Timedelta(minutes=15)).isoformat())
    manifest = dict(dataset="state_grid_1", cap_kw=100.)
    return frame, split, manifest


def test_future_observations_cannot_change_issue_features():
    frame, split, manifest = fixture_data()
    indices, future = sample_indices(frame, split, 64, 15)
    first = build_features(frame, indices, future, manifest, split)
    issue = indices["test"][0]
    altered = frame.copy()
    altered.loc[issue + 1:, ["power_kw", "tsi", "dni", "ghi", "temperature"]] += 500
    second = build_features(altered, indices, future, manifest, split)
    pos = np.flatnonzero(first[0] == issue)[0]
    for original, changed in zip(first[2], second[2]):
        np.testing.assert_array_equal(original[pos], changed[pos])
    np.testing.assert_array_equal(first[4][pos], second[4][pos])
    assert not np.array_equal(first[5][pos], second[5][pos])


def test_exact_horizons_and_gaps_and_partition_purge():
    frame, split, _ = fixture_data()
    frame = frame.drop(index=96 * 10).reset_index(drop=True)
    indices, future = sample_indices(frame, split, 64, 15)
    times = pd.DatetimeIndex(frame.ts)
    for name, idx in indices.items():
        for j, horizon in enumerate((15, 60, 240)):
            assert ((times[future[idx, j]] - times[idx]).total_seconds() / 60 == horizon).all()
        if name == "train":
            assert (times[future[idx, -1]] < pd.Timestamp(split["train_end"])).all()
        if name == "val":
            assert (times[future[idx, -1]] < pd.Timestamp(split["val_end"])).all()
        for i in idx:
            assert (np.diff(times[i - 63:i + 1].asi8) == 15 * 60 * 10**9).all()


def test_blend_is_convex_and_improves_complementary_errors():
    truth = np.array([0.2, 0.5, 0.8])
    w = blend_weights([truth + .1, truth - .1], truth)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.)
    np.testing.assert_allclose(w, [.5, .5], atol=1e-6)


def test_release_prediction_rejects_wrong_schema_and_preserves_quantiles():
    bundle = dict(feature_names=[["x"]] * 3, cap_kw=100., estimators={},
                  weights_per_horizon=[{"reference_baseline": 1.}] * 3,
                  residual_quantiles_kw=[[-10.] * 3, [0.] * 3, [10.] * 3])
    matrices = [np.zeros((2, 1))] * 3
    prediction, q = predict_feature_matrices(bundle, matrices, [["x"]] * 3, np.full((2, 3), .5))
    np.testing.assert_allclose(prediction, 50.)
    assert (np.diff(q, axis=1) >= 0).all()
    with pytest.raises(ValueError, match="feature order"):
        predict_feature_matrices(bundle, matrices, [["wrong"]] * 3, np.full((2, 3), .5))


def test_sky_hdf5_unsorted_source_indices_keep_causal_row_order(tmp_path, monkeypatch):
    monkeypatch.setattr(optimize, "ROOT", tmp_path)
    (tmp_path / "data/processed/skippd").mkdir(parents=True)
    source = tmp_path / "images.hdf5"
    images = np.stack([np.full((64, 64, 3), 51, dtype=np.uint8),
                       np.full((64, 64, 3), 204, dtype=np.uint8)])
    with h5py.File(source, "w") as stream:
        for group in ("trainval", "test"):
            stream.create_group(group).create_dataset("images_log", data=images)
    frame = pd.DataFrame(dict(image_group=["test", "trainval", "test", "trainval"],
                              image_index=[1, 1, 0, 0]))
    features, names = sky_descriptors(frame, source)
    np.testing.assert_allclose(features[:, names.index("image_mean_r")], [.8, .8, .2, .2], atol=5e-5)
    cached, _ = sky_descriptors(frame, source)
    np.testing.assert_array_equal(features, cached)
    reordered = frame.copy()
    reordered.image_group = reordered.image_group.iloc[::-1].to_numpy()
    with pytest.raises(ValueError, match="groups differ"):
        sky_descriptors(reordered, source)
