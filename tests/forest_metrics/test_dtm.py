import numpy as np

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import build_dtm, _grid_dtm, DTM


def _flat_ground_and_trees(ground_z=100.0, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 20, 40)
    ys = np.linspace(0, 20, 40)
    gx, gy = np.meshgrid(xs, ys)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, ground_z) + rng.normal(0, 0.01, gx.size)])

    tree_xy = rng.uniform(0, 20, size=(500, 2))
    tree_z = ground_z + rng.uniform(0.5, 15, size=500)
    trees = np.column_stack([tree_xy, tree_z])

    return ground, trees


def test_height_above_ground_zero_at_ground_level():
    ground, trees = _flat_ground_and_trees(ground_z=100.0)
    all_xyz = np.vstack([ground, trees])
    cfg = ForestMetricsConfig()

    grid_xyz, warnings = _grid_dtm(ground, all_xyz, cfg)
    dtm = DTM(grid_xyz)

    hag_ground = dtm.height_above_ground(ground)
    assert np.all(np.abs(hag_ground) < 0.1)


def test_height_above_ground_correct_offset_for_elevated_points():
    ground, trees = _flat_ground_and_trees(ground_z=100.0)
    all_xyz = np.vstack([ground, trees])
    cfg = ForestMetricsConfig()

    grid_xyz, _ = _grid_dtm(ground, all_xyz, cfg)
    dtm = DTM(grid_xyz)

    test_points = np.array([[10.0, 10.0, 105.0], [5.0, 5.0, 110.0]])
    hag = dtm.height_above_ground(test_points)
    assert abs(hag[0] - 5.0) < 0.2
    assert abs(hag[1] - 10.0) < 0.2


def test_build_dtm_no_ground_points_falls_back_without_raising():
    _, trees = _flat_ground_and_trees(ground_z=100.0)
    non_tree_xyz = np.zeros((0, 3))

    dtm, warnings = build_dtm(non_tree_xyz, trees, ForestMetricsConfig())

    assert len(warnings) > 0
    hag = dtm.height_above_ground(trees)
    assert np.all(np.isfinite(hag))


def test_convex_hull_area_positive_for_spread_out_grid():
    ground, trees = _flat_ground_and_trees(ground_z=100.0)
    all_xyz = np.vstack([ground, trees])
    grid_xyz, _ = _grid_dtm(ground, all_xyz, ForestMetricsConfig())
    dtm = DTM(grid_xyz)

    assert dtm.convex_hull_area_m2() > 0
