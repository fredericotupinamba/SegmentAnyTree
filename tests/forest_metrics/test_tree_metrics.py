import numpy as np
import pandas as pd

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import DTM
from tupisat_inference.forest_metrics.tree_metrics import (
    apply_monotonic_correction,
    fit_circle_robust,
    measure_tree,
    tilt_outlier_prob,
)


def _ring(xc, yc, r, n=200, noise=0.002, seed=0, arc_deg=360.0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.radians(arc_deg), n)
    x = xc + r * np.cos(theta) + rng.normal(0, noise, n)
    y = yc + r * np.sin(theta) + rng.normal(0, noise, n)
    return np.column_stack([x, y])


def test_fit_circle_robust_recovers_known_radius():
    cfg = ForestMetricsConfig()
    points = _ring(xc=10.0, yc=-5.0, r=0.15, n=300)

    xc, yc, r, cci = fit_circle_robust(points, cfg)

    assert np.isfinite(r)
    assert abs(r - 0.15) / 0.15 < 0.05
    assert abs(xc - 10.0) < 0.02
    assert abs(yc + 5.0) < 0.02
    assert 0 <= cci <= 1


def test_fit_circle_robust_too_few_points_returns_nan():
    cfg = ForestMetricsConfig()
    points = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

    xc, yc, r, cci = fit_circle_robust(points, cfg)

    assert np.isnan(r)
    assert np.isnan(xc)
    assert np.isnan(yc)
    assert np.isnan(cci)


def test_fit_circle_robust_empty_input_returns_nan():
    cfg = ForestMetricsConfig()
    points = np.zeros((0, 2))

    xc, yc, r, cci = fit_circle_robust(points, cfg)

    assert np.isnan(r)


def test_fit_circle_robust_recovers_stem_radius_despite_branch_noise():
    """The scenario that motivated porting dendromatics' clustering
    fallback: a clean stem ring plus a spatially separate cluster of
    'branch' noise points caught in the same horizontal slice. A plain
    least-squares fit over all points would be dragged off; fit_circle_check
    should isolate the larger, coherent ring cluster and recover its
    radius."""
    cfg = ForestMetricsConfig()
    ring_pts = _ring(xc=0.0, yc=0.0, r=0.15, n=250, noise=0.002)
    rng = np.random.default_rng(2)
    # A tight clump of "branch" points well outside the ring, on one side.
    branch_pts = np.column_stack(
        [rng.normal(0.5, 0.01, 40), rng.normal(0.5, 0.01, 40)]
    )
    points = np.vstack([ring_pts, branch_pts])

    xc, yc, r, cci = fit_circle_robust(points, cfg)

    assert np.isfinite(r)
    assert abs(r - 0.15) / 0.15 < 0.1
    assert abs(xc - 0.0) < 0.03
    assert abs(yc - 0.0) < 0.03


def test_tilt_outlier_prob_flags_sideways_center():
    heights = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    x_centers = np.zeros_like(heights)
    y_centers = np.zeros_like(heights)
    # One section's fitted center is pulled sideways, as if a branch
    # cluster dragged the fit off the tree's actual axis.
    y_centers[3] = 0.3
    radii = np.full_like(heights, 0.15)

    prob = tilt_outlier_prob(x_centers, y_centers, radii, heights)

    assert prob[3] > 0
    assert np.all(prob[[0, 1, 2, 4, 5, 6]] == 0)


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
    # ~280 degrees of angular coverage, uniformly at every height -> CCI
    # comfortably above the 0.3 fit-sanity floor (so dbh_cm is a real
    # number, not NaN) but below the stricter 0.8 tree-validation bar at
    # every single slice -- a plausible plot-edge/occlusion fragment, never
    # accumulating the min_high_cci_slices consistent good readings a real
    # trunk would. (dendromatics' sector_occupancy uses a fixed-width band
    # around the fitted radius rather than a radius-proportional ring, so
    # it's more sensitive to the fit bias a *very* partial arc -- e.g. 200
    # degrees -- introduces; 280 degrees keeps the fit itself trustworthy
    # while still failing the stricter validation bar.)
    tree_df = _cylinder_tree_df(radius_cm=15.0, z_top=12.0, arc_deg=280.0)

    row, _ = measure_tree(4, tree_df, _flat_dtm(), cfg)

    assert row["is_valid_tree"] is False
    assert np.isfinite(row["dbh_cm"])
    assert row["dbh_cci"] < cfg.tree_validation_cci_threshold


def test_measure_tree_accepts_tree_with_poor_coverage_at_a_single_height():
    """Real-data regression: three trees with excellent DBH CCI (0.85-1.0)
    were wrongly rejected by an earlier version of this check because their
    one dedicated slice at exactly tree_validation_height_m happened to have
    poor angular coverage (CCI 0.33-0.74) -- normal TLS/MLS coverage loss
    with height, not evidence of a shrub or fragment. Requiring several
    consistent good slices (rather than trusting a single height) should
    tolerate one bad slice on an otherwise well-formed tree."""
    cfg = ForestMetricsConfig()
    rng = np.random.default_rng(1)
    radius_m = 0.15
    z_top = 12.0
    zs = np.linspace(0.0, z_top, 200)
    rows = []
    for z in zs:
        arc_deg = 90.0 if abs(z - cfg.tree_validation_height_m) < 0.3 else 360.0
        theta = rng.uniform(0, np.radians(arc_deg), 150)
        x = radius_m * np.cos(theta)
        y = radius_m * np.sin(theta)
        rows.append(np.column_stack([x, y, np.full(150, z)]))
    tree_df = pd.DataFrame(np.vstack(rows), columns=["X", "Y", "Z"])

    row, _ = measure_tree(6, tree_df, _flat_dtm(), cfg)

    assert row["is_valid_tree"] is True
    assert "rejected_not_a_tree" not in row["quality_flags"]


def test_apply_monotonic_correction_interpolates_single_branch_bump():
    cfg = ForestMetricsConfig()
    heights = np.arange(1.6, 10.0, 0.5)
    diameters = 35.0 * (1 - heights / 14.0)  # smooth linear taper
    bump_idx = len(diameters) // 2
    diameters[bump_idx] += 15.0  # a branch inflates one mid-trunk sample
    taper_df = pd.DataFrame({"height_m": heights, "diameter_cm": diameters, "n_points": 200})

    corrected = apply_monotonic_correction(taper_df, cfg)

    values = corrected["diameter_corrected_cm"].values
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) <= 1e-6)  # non-increasing
    # the bump was replaced by interpolation between its neighbours, not
    # left untouched, and not just clamped down to one of them
    neighbour_avg = (values[bump_idx - 1] + values[bump_idx + 1]) / 2
    assert abs(values[bump_idx] - neighbour_avg) < 0.5
    assert values[bump_idx] < diameters[bump_idx]


def test_apply_monotonic_correction_does_not_let_a_bad_run_affect_clean_samples_below():
    """The whole point of the local rule over a global fit: a run of bad
    (increasing) samples above a clean, decreasing run must not change any
    of the clean values -- this is exactly what an earlier isotonic-
    regression version of this function got wrong on real data, collapsing
    an entire tree's corrected diameters (including untouched clean low
    samples) to a single flat, physically implausible number."""
    cfg = ForestMetricsConfig()
    heights = np.arange(0.1, 8.0, 0.5)
    clean_diameters = 31.0 * (1 - heights / 20.0)  # smooth, decreasing taper

    garbage_from = len(heights) // 2
    diameters = clean_diameters.copy()
    diameters[garbage_from:] = [125.2, 78.5, 158.6, 35.8, 125.2, 144.2, 120.2, 139.7, 236.8, 90.0, 60.0][
        : len(heights) - garbage_from
    ]

    taper_df = pd.DataFrame({"height_m": heights, "diameter_cm": diameters, "n_points": 200})

    corrected = apply_monotonic_correction(taper_df, cfg)

    clean_corrected = corrected["diameter_corrected_cm"].values[:garbage_from]
    assert np.array_equal(clean_corrected, clean_diameters[:garbage_from])


def test_apply_monotonic_correction_flags_value_above_min_of_last_two():
    """Directly exercises the stated rule: flag (and interpolate) a sample
    only when it exceeds the smaller of the two accepted samples below it,
    not just any local increase."""
    cfg = ForestMetricsConfig()
    heights = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    diameters = np.array([20.0, 18.0, 40.0, 17.0, 16.0])  # index 2 is the violator
    taper_df = pd.DataFrame({"height_m": heights, "diameter_cm": diameters, "n_points": 200})

    corrected = apply_monotonic_correction(taper_df, cfg)
    values = corrected["diameter_corrected_cm"].values

    assert values[0] == 20.0 and values[1] == 18.0  # untouched, nothing to compare against yet
    assert values[2] != 40.0  # flagged and interpolated
    assert values[3] == 17.0 and values[4] == 16.0  # untouched, not violators


def test_apply_monotonic_correction_output_never_increases():
    """The min-of-last-two reference (rather than their average) is what
    guarantees this: an accepted value can never exceed the accepted value
    right before it, so small upward creep across many small, individually
    tolerable-looking steps can't accumulate the way it would with an
    average-based reference."""
    cfg = ForestMetricsConfig()
    heights = np.arange(0.1, 12.0, 0.2)
    rng = np.random.default_rng(3)
    # A gently decreasing trend with enough noise that a naive
    # average-of-last-two reference lets the curve creep upward over many
    # small steps (verified this exact data drifted up under that rule).
    diameters = 30.0 * (1 - heights / 16.0) + rng.normal(0, 0.6, heights.shape)
    taper_df = pd.DataFrame({"height_m": heights, "diameter_cm": diameters, "n_points": 200})

    corrected = apply_monotonic_correction(taper_df, cfg)
    values = corrected["diameter_corrected_cm"].values

    assert np.all(np.diff(values) <= 1e-9)


def test_apply_monotonic_correction_leaves_already_decreasing_curve_untouched():
    cfg = ForestMetricsConfig()
    heights = np.arange(1.6, 10.0, 0.5)
    diameters = 35.0 * (1 - heights / 14.0)
    taper_df = pd.DataFrame({"height_m": heights, "diameter_cm": diameters, "n_points": 200})

    corrected = apply_monotonic_correction(taper_df, cfg)

    assert np.array_equal(corrected["diameter_corrected_cm"].values, diameters)


def test_apply_monotonic_correction_handles_empty_taper():
    cfg = ForestMetricsConfig()
    empty = pd.DataFrame(columns=["height_m", "diameter_cm", "n_points"])

    corrected = apply_monotonic_correction(empty, cfg)

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
