import numpy as np
import pandas as pd

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import DTM
from tupisat_inference.forest_metrics.tree_metrics import (
    apply_monotonic_correction,
    fit_circle_ransac,
    measure_tree,
)


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


def _flat_dtm(z=0.0, extent=5.0, step=0.5):
    xs = np.arange(-extent, extent + step, step)
    ys = np.arange(-extent, extent + step, step)
    gx, gy = np.meshgrid(xs, ys)
    grid_xyz = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
    return DTM(grid_xyz)


def _cylinder_tree_df(radius_cm, z_top, z_bottom=0.0, arc_deg=360.0, points_per_ring=150,
                       ring_spacing_m=0.05, seed=0):
    """A vertical cylindrical shell of points -- a stand-in for a scanned
    stem. arc_deg < 360 simulates a partially-occluded/plot-edge scan (only
    that many degrees of the circumference were ever hit)."""
    rng = np.random.default_rng(seed)
    radius_m = radius_cm / 100.0
    n_rings = max(int((z_top - z_bottom) / ring_spacing_m), 2)
    zs = np.linspace(z_bottom, z_top, n_rings)
    rows = []
    for z in zs:
        theta = rng.uniform(0, np.radians(arc_deg), points_per_ring)
        x = radius_m * np.cos(theta)
        y = radius_m * np.sin(theta)
        rows.append(np.column_stack([x, y, np.full(points_per_ring, z)]))
    xyz = np.vstack(rows)
    return pd.DataFrame(xyz, columns=["X", "Y", "Z"])


def test_measure_tree_accepts_well_formed_tall_tree():
    cfg = ForestMetricsConfig()
    tree_df = _cylinder_tree_df(radius_cm=15.0, z_top=12.0)  # DBH ~30cm, full ring

    row, taper_df = measure_tree(1, tree_df, _flat_dtm(), cfg)

    assert row["is_valid_tree"] is True
    assert abs(row["dbh_cm"] - 30.0) / 30.0 < 0.1
    assert "rejected_not_a_tree" not in row["quality_flags"]


def test_measure_tree_rejects_short_shrub():
    cfg = ForestMetricsConfig()
    # Full, dense ring at breast height, but nothing left by 4m -- exactly
    # the shrub/undergrowth case tree_validation_height_m targets.
    tree_df = _cylinder_tree_df(radius_cm=15.0, z_top=2.5)

    row, _ = measure_tree(2, tree_df, _flat_dtm(), cfg)

    assert row["is_valid_tree"] is False
    assert "rejected_not_a_tree" in row["quality_flags"]


def test_measure_tree_rejects_stem_thinner_than_min_valid_dbh():
    cfg = ForestMetricsConfig()
    tree_df = _cylinder_tree_df(radius_cm=2.5, z_top=12.0)  # DBH ~5cm < 7cm

    row, _ = measure_tree(3, tree_df, _flat_dtm(), cfg)

    assert row["is_valid_tree"] is False
    assert row["dbh_cm"] < cfg.min_valid_dbh_cm


def test_measure_tree_rejects_partial_arc_plot_edge_fragment():
    cfg = ForestMetricsConfig()
    # ~200 degrees of angular coverage -> CCI comfortably above the 0.3
    # fit-sanity floor (so dbh_cm is a real number, not NaN) but below the
    # stricter 0.8 tree-validation bar -- a plausible plot-edge/occlusion
    # fragment rather than an outright failed fit.
    tree_df = _cylinder_tree_df(radius_cm=15.0, z_top=12.0, arc_deg=200.0)

    row, _ = measure_tree(4, tree_df, _flat_dtm(), cfg)

    assert row["is_valid_tree"] is False
    assert np.isfinite(row["dbh_cm"])
    assert row["dbh_cci"] < cfg.tree_validation_cci_threshold


def _taper_df_with_branch_bump(seed=0):
    heights = np.arange(1.6, 10.0, 0.5)
    diameters = 35.0 * (1 - heights / 14.0)  # smooth linear taper
    bump_idx = len(diameters) // 2
    diameters[bump_idx] += 15.0  # a branch inflates one mid-trunk sample
    return pd.DataFrame({
        "height_m": heights,
        "diameter_cm": diameters,
        "n_points": 200,
    }), bump_idx


def test_apply_monotonic_correction_pulls_down_branch_bump():
    cfg = ForestMetricsConfig()
    taper_df, bump_idx = _taper_df_with_branch_bump()

    corrected = apply_monotonic_correction(taper_df, dbh_cm=38.0, dbh_height_m=1.3, dbh_n_points=2000, cfg=cfg)

    values = corrected["diameter_corrected_cm"].values
    assert np.all(np.isfinite(values))
    # non-increasing (isotonic regression guarantees this up to float noise)
    assert np.all(np.diff(values) <= 1e-6)
    # the bump was pulled down towards its neighbours, not left untouched
    assert values[bump_idx] < taper_df["diameter_cm"].values[bump_idx]


def test_apply_monotonic_correction_leaves_already_monotonic_curve_close_to_raw():
    cfg = ForestMetricsConfig()
    heights = np.arange(1.6, 10.0, 0.5)
    diameters = 35.0 * (1 - heights / 14.0)
    taper_df = pd.DataFrame({"height_m": heights, "diameter_cm": diameters, "n_points": 200})

    corrected = apply_monotonic_correction(taper_df, dbh_cm=38.0, dbh_height_m=1.3, dbh_n_points=2000, cfg=cfg)

    values = corrected["diameter_corrected_cm"].values
    assert np.all(np.diff(values) <= 1e-6)
    assert np.allclose(values, diameters, atol=2.0)


def test_apply_monotonic_correction_handles_empty_taper():
    cfg = ForestMetricsConfig()
    empty = pd.DataFrame(columns=["height_m", "diameter_cm", "n_points"])

    corrected = apply_monotonic_correction(empty, dbh_cm=np.nan, dbh_height_m=1.3, dbh_n_points=0, cfg=cfg)

    assert "diameter_corrected_cm" in corrected.columns
    assert corrected.empty


def test_measure_tree_taper_stops_below_full_height_when_crown_detected():
    cfg = ForestMetricsConfig()
    rng = np.random.default_rng(0)

    trunk = _cylinder_tree_df(radius_cm=15.0, z_top=6.0, points_per_ring=30, ring_spacing_m=0.1)
    # A much denser, wider scatter above the trunk -- stands in for a real
    # crown's higher point density (foliage/branch returns), which is what
    # the crown_base_height_m heuristic actually keys off of.
    n_crown = 6000
    crown_xy = rng.uniform(-1.5, 1.5, size=(n_crown, 2))
    crown_z = rng.uniform(6.0, 10.0, size=n_crown)
    crown = pd.DataFrame(np.column_stack([crown_xy, crown_z]), columns=["X", "Y", "Z"])
    tree_df = pd.concat([trunk, crown], ignore_index=True)

    row, taper_df = measure_tree(5, tree_df, _flat_dtm(), cfg)

    assert row["crown_base_height_m"] < row["height_m"] - 1.0
    assert taper_df.empty or taper_df["height_m"].max() <= row["crown_base_height_m"] + cfg.taper_height_increment_m
