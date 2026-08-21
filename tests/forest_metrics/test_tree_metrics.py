import numpy as np

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.tree_metrics import fit_circle_ransac


def _ring(xc, yc, r, n=200, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    x = xc + r * np.cos(theta) + rng.normal(0, noise, n)
    y = yc + r * np.sin(theta) + rng.normal(0, noise, n)
    return np.column_stack([x, y])


def test_fit_circle_ransac_recovers_known_radius():
    cfg = ForestMetricsConfig()
    points = _ring(xc=10.0, yc=-5.0, r=0.15, n=300)

    xc, yc, r, cci = fit_circle_ransac(points, cfg, max_trials=cfg.dbh_ransac_max_trials)

    assert np.isfinite(r)
    assert abs(r - 0.15) / 0.15 < 0.05
    assert abs(xc - 10.0) < 0.02
    assert abs(yc + 5.0) < 0.02
    assert 0 <= cci <= 1


def test_fit_circle_ransac_too_few_points_returns_nan():
    cfg = ForestMetricsConfig()
    points = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

    xc, yc, r, cci = fit_circle_ransac(points, cfg, max_trials=cfg.dbh_ransac_max_trials)

    assert np.isnan(r)
    assert np.isnan(xc)
    assert np.isnan(yc)
    assert np.isnan(cci)


def test_fit_circle_ransac_empty_input_returns_nan():
    cfg = ForestMetricsConfig()
    points = np.zeros((0, 2))

    xc, yc, r, cci = fit_circle_ransac(points, cfg, max_trials=cfg.dbh_ransac_max_trials)

    assert np.isnan(r)
