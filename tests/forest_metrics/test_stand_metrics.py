import numpy as np
import pandas as pd

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import DTM
from tupisat_inference.forest_metrics.stand_metrics import (
    compute_spacing_indices,
    summarize_plot,
)


def _flat_dtm(size=50.0, z=100.0):
    grid = np.array([[0, 0, z], [size, 0, z], [0, size, z], [size, size, z]])
    return DTM(grid)


def test_clark_evans_regular_grid_is_greater_than_one():
    xs, ys = np.meshgrid(np.arange(0, 40, 5), np.arange(0, 40, 5))
    bases = pd.DataFrame({"x_base": xs.ravel(), "y_base": ys.ravel()})
    plot_area_ha = (40 * 40) / 10000

    result = compute_spacing_indices(bases, plot_area_ha)

    assert result["clark_evans_r"] > 1


def test_clark_evans_clustered_points_is_less_than_one():
    rng = np.random.default_rng(0)
    cluster_centers = np.array([[5, 5], [30, 30], [5, 30]])
    points = []
    for cx, cy in cluster_centers:
        points.append(rng.normal([cx, cy], 0.3, size=(10, 2)))
    points = np.vstack(points)
    bases = pd.DataFrame({"x_base": points[:, 0], "y_base": points[:, 1]})
    plot_area_ha = (40 * 40) / 10000

    result = compute_spacing_indices(bases, plot_area_ha)

    assert result["clark_evans_r"] < 1


def test_spacing_indices_nan_for_fewer_than_two_trees():
    bases = pd.DataFrame({"x_base": [1.0], "y_base": [1.0]})
    result = compute_spacing_indices(bases, plot_area_ha=1.0)
    assert np.isnan(result["clark_evans_r"])
    assert np.isnan(result["mean_nn_distance_m"])


def test_summarize_plot_zero_trees_does_not_raise():
    dtm = _flat_dtm()
    empty_tree_metrics = pd.DataFrame(columns=["dbh_cm", "x_base", "y_base"])
    empty_tree_pts = pd.DataFrame(columns=["x", "y", "z"])

    summary = summarize_plot(empty_tree_metrics, dtm, empty_tree_pts, ForestMetricsConfig())

    assert summary["n_trees"] == 0
    assert summary["plot_area_ha"] > 0
    assert np.isnan(summary["basal_area_m2_ha"])
