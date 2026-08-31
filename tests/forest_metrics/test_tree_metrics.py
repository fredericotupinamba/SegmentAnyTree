import numpy as np
import pandas as pd

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import DTM
from tupisat_inference.forest_metrics.tree_metrics import (
    apply_monotonic_correction,
    axis_center_at,
    fit_circle_cylinder_window,
    fit_circle_near_axis,
    fit_circle_robust,
    fit_stem_axis,
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
    df = pd.DataFrame(xyz, columns=["X", "Y", "Z"])
    df["intensity"] = 10000.0  # plausible constant bark-like intensity; irrelevant to these tests' assertions
    return df


def test_measure_tree_accepts_well_formed_tall_tree():
    cfg = ForestMetricsConfig()
    tree_df = _cylinder_tree_df(radius_cm=15.0, z_top=12.0)  # DBH ~30cm, full ring

    row, taper_df, _ = measure_tree(1, tree_df, _flat_dtm(), cfg)

    assert row["is_valid_tree"] is True
    assert abs(row["dbh_cm"] - 30.0) / 30.0 < 0.1
    assert "rejected_not_a_tree" not in row["quality_flags"]


def test_measure_tree_rejects_short_shrub():
    cfg = ForestMetricsConfig()
    # Full, dense ring at breast height, but nothing left by 4m -- exactly
    # the shrub/undergrowth case tree_validation_height_m targets.
    tree_df = _cylinder_tree_df(radius_cm=15.0, z_top=2.5)

    row, _, _ = measure_tree(2, tree_df, _flat_dtm(), cfg)

    assert row["is_valid_tree"] is False
    assert "rejected_not_a_tree" in row["quality_flags"]


def test_measure_tree_rejects_stem_thinner_than_min_valid_dbh():
    cfg = ForestMetricsConfig()
    tree_df = _cylinder_tree_df(radius_cm=2.5, z_top=12.0)  # DBH ~5cm < 7cm

    row, _, _ = measure_tree(3, tree_df, _flat_dtm(), cfg)

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

    row, _, _ = measure_tree(4, tree_df, _flat_dtm(), cfg)

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
    tree_df["intensity"] = 10000.0

    row, _, _ = measure_tree(6, tree_df, _flat_dtm(), cfg)

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
    """compute_crown_metrics keys off LiDAR intensity dropping and
    horizontal footprint area growing relative to the tree's own trunk
    baseline (calibrated against real data -- see config.py). The
    synthetic crown here is both much wider (a scatter over ~3x3m vs. the
    trunk's ~0.3m-diameter ring) and much lower-intensity than the trunk,
    matching the real-world pattern (foliage returns read far dimmer than
    bark in LiDAR intensity)."""
    cfg = ForestMetricsConfig()
    rng = np.random.default_rng(0)

    # A clear height gap between the trunk top (5.5m) and crown bottom
    # (6.5m) keeps the "trunk points never flagged as crown" assertion
    # below unambiguous regardless of exactly which bin edge the detector
    # picks as the transition.
    trunk = _cylinder_tree_df(radius_cm=15.0, z_top=5.5, points_per_ring=30, ring_spacing_m=0.1)
    n_crown = 6000
    crown_xy = rng.uniform(-1.5, 1.5, size=(n_crown, 2))
    crown_z = rng.uniform(6.5, 10.0, size=n_crown)
    crown = pd.DataFrame(np.column_stack([crown_xy, crown_z]), columns=["X", "Y", "Z"])
    crown["intensity"] = 3000.0  # well below trunk's 10000.0 -- crown_intensity_ratio_threshold is 0.4
    tree_df = pd.concat([trunk, crown], ignore_index=True)

    row, taper_df, is_crown_point = measure_tree(5, tree_df, _flat_dtm(), cfg)

    assert row["crown_base_height_m"] < row["height_m"] - 1.0
    assert taper_df.empty or taper_df["height_m"].max() <= row["crown_base_height_m"] + cfg.taper_height_increment_m
    assert is_crown_point.shape[0] == len(tree_df)
    assert not is_crown_point[: len(trunk)].any()  # trunk points never flagged as crown
    assert is_crown_point[len(trunk):].mean() > 0.5  # most of the synthetic crown scatter is flagged


def _wood_leaf_tree_df(trunk_top_m=8.0, tree_top_m=14.0, seed=0):
    """A bole of pure wood points topped by a leaf-dominated crown -- the
    real pattern the wood-fraction rule keys off. On real calibration data
    a bare bole bins at ~99.9% wood and a crown bin at ~0% (see config.py),
    so the crown here is given a small wood minority (branches) rather than
    none, to keep the fixture honest about what the rule must tolerate."""
    rng = np.random.default_rng(seed)
    trunk = _cylinder_tree_df(radius_cm=15.0, z_top=trunk_top_m, points_per_ring=40, ring_spacing_m=0.05)
    trunk["prediction"] = 1.0

    n_crown = 8000
    crown = pd.DataFrame(
        np.column_stack([
            rng.uniform(-1.5, 1.5, n_crown),
            rng.uniform(-1.5, 1.5, n_crown),
            rng.uniform(trunk_top_m, tree_top_m, n_crown),
        ]),
        columns=["X", "Y", "Z"],
    )
    crown["intensity"] = 10000.0  # same as the trunk: only the wood/leaf label may drive the split
    crown["prediction"] = (rng.random(n_crown) < 0.15).astype(float)
    return pd.concat([trunk, crown], ignore_index=True), len(trunk)


def test_crown_base_uses_wood_fraction_when_prediction_present():
    """With a wood/leaf label the crown base comes from the wood fraction
    alone. Intensity is deliberately identical in trunk and crown here, so
    the intensity+area fallback could not find this transition at all --
    only the wood label can."""
    cfg = ForestMetricsConfig()
    tree_df, n_trunk = _wood_leaf_tree_df(trunk_top_m=8.0, tree_top_m=14.0)

    row, _taper_df, is_crown_point = measure_tree(1, tree_df, _flat_dtm(), cfg)

    # The rule fires where wood drops off (8.0m) and adds its fitted
    # offset, so the answer sits near 8.0 + crown_wood_offset_m.
    assert abs(row["crown_base_height_m"] - (8.0 + cfg.crown_wood_offset_m)) < 1.0
    assert is_crown_point.shape[0] == len(tree_df)
    assert not is_crown_point[:n_trunk].any()


def test_crown_base_falls_back_when_prediction_absent():
    """Same geometry without the label: the wood rule cannot run, so the
    intensity+area fallback takes over. It finds no transition on this
    constant-intensity fixture, which is the honest outcome -- reporting
    NaN rather than guessing (measure_tree then uses the full height)."""
    cfg = ForestMetricsConfig()
    tree_df, _ = _wood_leaf_tree_df(trunk_top_m=8.0, tree_top_m=14.0)
    tree_df = tree_df.drop(columns=["prediction"])

    row, _taper_df, is_crown_point = measure_tree(1, tree_df, _flat_dtm(), cfg)

    assert np.isnan(row["crown_base_height_m"])
    assert not is_crown_point.any()


def test_crown_base_offset_cannot_exceed_tree_height():
    """The fitted offset is added to the run's start height, so on a short
    tree whose wood gives out near the top it could otherwise land above
    the treetop -- crown_base_height_m must stay inside the tree."""
    cfg = ForestMetricsConfig()
    tree_df, _ = _wood_leaf_tree_df(trunk_top_m=5.0, tree_top_m=5.6)

    row, _taper_df, _is_crown = measure_tree(1, tree_df, _flat_dtm(), cfg)

    if np.isfinite(row["crown_base_height_m"]):
        assert row["crown_base_height_m"] <= row["height_m"]


def _leaning_sections(lean_dx=0.05, lean_dy=0.02, radius_m=0.15, n=20):
    """Section centres along a leaning stem, plus their radii."""
    heights = np.arange(0.1, 0.1 + 0.2 * n, 0.2)
    cx = 100.0 + lean_dx * heights
    cy = 200.0 + lean_dy * heights
    return heights, cx, cy, np.full(heights.shape, radius_m)


def test_fit_stem_axis_recovers_lean():
    """The axis must be fitted, not assumed vertical: most real stems lean
    (median 3 degrees on the calibration plots), and a vertical assumption
    would read every section of a leaning tree as off-axis."""
    cfg = ForestMetricsConfig()
    heights, cx, cy, radii = _leaning_sections(lean_dx=0.05, lean_dy=0.02)

    axis = fit_stem_axis(heights, cx, cy, radii, cfg)

    assert axis is not None
    assert abs(axis["dx"] - 0.05) < 1e-6
    assert abs(axis["dy"] - 0.02) < 1e-6
    assert axis["on_axis"].all()
    px, py = axis_center_at(axis, 5.0)
    assert abs(px - (100.0 + 0.05 * 5.0)) < 1e-3
    assert abs(py - (200.0 + 0.02 * 5.0)) < 1e-3


def test_fit_stem_axis_ignores_sections_captured_by_a_branch():
    """A few sections centred on a branch must not drag the axis -- that is
    the whole reason for a robust (Theil-Sen) fit over least squares."""
    cfg = ForestMetricsConfig()
    heights, cx, cy, radii = _leaning_sections()
    cx[12:15] += 0.6  # three sections pulled onto a branch

    axis = fit_stem_axis(heights, cx, cy, radii, cfg)

    assert axis is not None
    assert abs(axis["dx"] - 0.05) < 1e-3
    assert not axis["on_axis"][12:15].any()


def test_fit_circle_near_axis_excludes_branch_cluster():
    """The failure this whole path exists for: a branch beside the stem in
    the same slice. Unconstrained, the fit spans both and reads far too
    large; constrained to the axis window it recovers the stem."""
    cfg = ForestMetricsConfig()
    stem = _ring(100.0, 200.0, 0.14, n=260)
    branch = _ring(100.42, 200.0, 0.10, n=200, seed=3)
    # A bridge of points joining the two: a branch is attached to the stem,
    # so single-linkage clustering (fit_circle_check's existing fallback)
    # cannot split them. With a detached branch that fallback already
    # succeeds, and this path would never be needed.
    bridge = np.column_stack([
        np.linspace(100.13, 100.33, 40),
        np.full(40, 200.0) + np.random.default_rng(7).normal(0, 0.004, 40),
    ])
    both = np.vstack([stem, branch, bridge])

    _, _, r_free, _ = fit_circle_robust(both, cfg)
    _, _, r_axis, _ = fit_circle_near_axis(both, (100.0, 200.0), 0.14, cfg)

    assert np.isfinite(r_axis)
    assert abs(r_axis - 0.14) < 0.02
    # The unconstrained fit is the one being repaired: it must not already
    # be right, or this test would prove nothing.
    assert not np.isfinite(r_free) or abs(r_free - 0.14) > abs(r_axis - 0.14)


def test_cylinder_window_rejects_a_branch_that_does_not_span_it():
    """Layer 3's discriminator: stem points fill the vertical window, a
    branch crossing it occupies only a thin band of height. Same points in
    plan view, different answer."""
    cfg = ForestMetricsConfig()
    axis = {"x0": 100.0, "y0": 200.0, "dx": 0.0, "dy": 0.0}
    rng = np.random.default_rng(0)

    ring = _ring(100.0, 200.0, 0.14, n=400)
    stem_z = rng.uniform(4.7, 5.3, ring.shape[0])  # spans the whole window
    stem = np.column_stack([ring, stem_z])

    ring_b = _ring(100.0, 200.0, 0.14, n=400, seed=5)
    flat_z = rng.uniform(4.98, 5.02, ring_b.shape[0])  # a thin slab only
    slab = np.column_stack([ring_b, flat_z])

    _, _, r_stem, _ = fit_circle_cylinder_window(
        stem, stem[:, 2], 5.0, axis, 0.14, cfg)
    _, _, r_slab, _ = fit_circle_cylinder_window(
        slab, slab[:, 2], 5.0, axis, 0.14, cfg)

    assert np.isfinite(r_stem) and abs(r_stem - 0.14) < 0.02
    assert not np.isfinite(r_slab)


def test_axis_refit_never_reports_a_clipped_stem():
    """A refit whose window missed the stem returns a small arc, which fits
    a small circle very convincingly. Accepting that would replace a value
    that is too large with one that is too small -- worse, because it looks
    plausible. The plausibility band must reject it."""
    cfg = ForestMetricsConfig()
    stem = _ring(100.0, 200.0, 0.20, n=400)

    # Predicted centre 0.30 m off the true stem: the window catches an arc.
    _, _, r, _ = fit_circle_near_axis(stem, (100.30, 200.0), 0.20, cfg)

    if np.isfinite(r):
        assert not (cfg.axis_refit_min_radius_frac * 0.20 <= r
                     <= cfg.axis_refit_max_radius_frac * 0.20), (
            "a clipped-stem fit landed inside the plausibility band")
