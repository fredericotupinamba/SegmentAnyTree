from typing import Tuple

import numpy as np
import pandas as pd
from scipy import optimize as opt
from scipy.cluster import hierarchy as sch
from scipy.spatial import ConvexHull, distance_matrix

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import DTM

NAN_CIRCLE = (np.nan, np.nan, np.nan, np.nan)


# -----------------------------------------------------------------------------
# Circle fitting, ported from 3DFin/dendromatics' dendromatics/sections.py
# (fit_circle, inner_circle, sector_occupancy, point_clustering,
# fit_circle_check). Pure numpy/scipy, no new dependency. Replaces an
# earlier RANSAC-based fit (skimage CircleModel) that had no equivalent of
# fit_circle_check's spatial-clustering fallback -- a branch/foliage
# cluster caught in the same horizontal slice as the stem could pull a
# RANSAC fit off badly depending on tuning, with no recovery path.
# -----------------------------------------------------------------------------


def _lsq_fit_circle(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares circle fit (dendromatics.sections.fit_circle)."""

    def _radii(xc, yc):
        return np.hypot(X - xc, Y - yc)

    def _residuals(c):
        r = _radii(*c)
        return r - r.mean()

    center, _ = opt.leastsq(_residuals, (X.mean(), Y.mean()), maxfev=2000)
    radius = _radii(*center).mean()
    return center[0], center[1], radius


def _inner_circle_point_count(X: np.ndarray, Y: np.ndarray, xc: float, yc: float, r: float, times_r: float) -> int:
    """Points falling inside a circle shrunk by `times_r` -- a good fit
    should have its points on the ring, not bunched near the center
    (dendromatics.sections.inner_circle)."""
    dist = np.hypot(X - xc, Y - yc)
    return int(np.sum(dist < r * times_r))


def _sector_occupancy(
    X: np.ndarray, Y: np.ndarray, xc: float, yc: float, r: float, n_sectors: int, min_n_sectors: int, width: float
) -> Tuple[float, bool]:
    """Percentage of angular sectors around the fitted circle that contain
    a point within `width` of the fitted radius, and whether that meets
    `min_n_sectors` (dendromatics.sections.sector_occupancy)."""
    dx, dy = X - xc, Y - yc
    radial = np.hypot(dx, dy)
    angular = np.arctan2(dx, dy)

    within_band = (radial > (r - width)) & (radial < (r + width))
    sector_idx = np.floor(angular[within_band] / (2 * np.pi / n_sectors))
    n_occupied = np.unique(sector_idx).size

    perct_occupied = n_occupied * 100 / n_sectors
    return perct_occupied, n_occupied >= min_n_sectors


def _largest_point_cluster(X: np.ndarray, Y: np.ndarray, max_dist: float) -> Tuple[np.ndarray, np.ndarray]:
    """Largest spatially-coherent cluster of points, used as a fallback
    when the initial fit fails quality checks -- isolates the stem ring
    from e.g. a branch/foliage cluster caught in the same slice
    (dendromatics.sections.point_clustering)."""
    xy = np.column_stack((X, Y))
    cluster_id = sch.fclusterdata(xy, max_dist, criterion="distance", metric="euclidean")
    counts = np.bincount(cluster_id)
    largest = np.argmax(counts)
    mask = cluster_id == largest
    return X[mask], Y[mask]


def _fit_circle_check(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: ForestMetricsConfig,
    allow_cluster_fallback: bool = True,
) -> Tuple[float, float, float, float, int]:
    """Fits a circle and validates it via inner-circle + sector-occupancy
    checks; on failure, retries once on the largest spatial point cluster
    (dendromatics.sections.fit_circle_check, recursive fallback capped at
    one retry). Returns (xc, yc, r, sector_perct, n_points_in); r is 0 if
    no valid fit could be produced."""
    if X.size <= cfg.min_points_for_circle_fit:
        return 0.0, 0.0, 0.0, 0.0, 0

    xc, yc, r = _lsq_fit_circle(X, Y)
    n_points_in = _inner_circle_point_count(X, Y, xc, yc, r, cfg.circle_diameter_proportion)
    sector_perct, enough_sectors = _sector_occupancy(
        X, Y, xc, yc, r, cfg.circle_n_sectors, cfg.circle_min_occupied_sectors, cfg.circle_sector_width_m
    )

    fit_is_bad = (
        n_points_in > cfg.circle_inner_points_threshold
        or r < cfg.circle_min_radius_m
        or 2 * r * 100 > cfg.max_plausible_diameter_cm
        or not enough_sectors
    )
    if fit_is_bad and allow_cluster_fallback:
        X_g, Y_g = _largest_point_cluster(X, Y, cfg.circle_max_point_distance_m)
        if X_g.size > cfg.min_points_for_circle_fit:
            return _fit_circle_check(X_g, Y_g, cfg, allow_cluster_fallback=False)
        return 0.0, 0.0, 0.0, 0.0, 0

    return xc, yc, r, sector_perct, n_points_in


def fit_circle_robust(points_xy: np.ndarray, cfg: ForestMetricsConfig) -> Tuple[float, float, float, float]:
    """Fit a circle to a horizontal slice of points via the dendromatics
    fit_circle_check pipeline. Returns (xc, yc, radius_m, cci); all NaN if
    there aren't enough points or no valid fit could be produced. `cci` is
    the fraction (0-1) of angular sectors around the ring that contain a
    point -- same role and scale as the earlier ring-based CCI."""
    n = points_xy.shape[0]
    if n < cfg.min_points_for_circle_fit:
        return NAN_CIRCLE

    # points_xy arrives in real-world UTM-scale coordinates (6-7 digit
    # values); centering on the centroid before fitting avoids the
    # numerical instability plain least-squares circle fits can hit at
    # that magnitude (verified against real data with the previous
    # skimage-based fit -- see git history).
    centroid = points_xy.mean(axis=0)
    X = points_xy[:, 0] - centroid[0]
    Y = points_xy[:, 1] - centroid[1]

    xc, yc, r, sector_perct, _ = _fit_circle_check(X, Y, cfg)
    if not np.isfinite(r) or r <= 0 or 2 * r * 100 > cfg.max_plausible_diameter_cm:
        return NAN_CIRCLE

    return xc + centroid[0], yc + centroid[1], r, sector_perct / 100


# -----------------------------------------------------------------------------
# tilt_detection, ported from dendromatics/sections.py and adapted to run on
# one tree's sections at a time (dendromatics batches all trees into 2D
# matrices; measure_tree already processes one tree per call). Flags a
# section whose fitted circle center deviates from the tree's other section
# centers -- a fit "pulled sideways" (e.g. by a branch cluster) can still
# pass the radius/coverage checks above while being centered on the wrong
# spot.
# -----------------------------------------------------------------------------


def _outlier_flags(values: np.ndarray, n_range: float = 1.5) -> np.ndarray:
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr * n_range, q3 + iqr * n_range
    return ((values < lower) | (values > upper)).astype(float)


def tilt_outlier_prob(
    x_centers: np.ndarray, y_centers: np.ndarray, radii: np.ndarray, heights: np.ndarray, w_1: float = 3.0, w_2: float = 1.0
) -> np.ndarray:
    """Outlier score (0-1) per section: a weighted sum of how tilted a
    section's center is relative to *all* other section centers (absolute)
    and relative to each individual other section (relative). Sections with
    radius <= 0 (no valid fit) always score 0."""
    outlier_prob = np.zeros_like(x_centers, dtype=float)
    valid = radii > 0
    n_valid = int(np.sum(valid))
    if n_valid < 3:
        return outlier_prob

    abs_w = w_1 / (n_valid * w_2 + w_1)
    rel_w = w_2 / (n_valid * w_2 + w_1)

    h = heights[valid].reshape(-1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_dist = distance_matrix(h, h)
        xy_dist = distance_matrix(np.column_stack((x_centers[valid], y_centers[valid])), np.column_stack((x_centers[valid], y_centers[valid])))
        tilt_matrix = np.degrees(np.arctan(xy_dist / z_dist))

    valid_idx = np.flatnonzero(valid)
    outlier_prob[valid_idx] = _outlier_flags(np.nansum(tilt_matrix, axis=0)) * abs_w

    others_mask = ~np.eye(n_valid, dtype=bool)
    for j in range(n_valid):
        row_excl_self = tilt_matrix[j][others_mask[j]]
        row = np.where(np.arange(n_valid) == j, np.median(row_excl_self), tilt_matrix[j])
        outlier_prob[valid_idx] += _outlier_flags(row) * rel_w

    return outlier_prob


def compute_tree_base(tree_xyz: np.ndarray, hag: np.ndarray, dtm: DTM, cfg: ForestMetricsConfig) -> Tuple[float, float, float]:
    base_mask = hag <= cfg.base_slice_thickness_m
    if not np.any(base_mask):
        # No low points at all (e.g. plot-edge or canopy-only capture) --
        # fall back to the tree's lowest points instead of failing outright.
        base_mask = hag <= np.nanpercentile(hag, 5)
        if not np.any(base_mask):
            return np.nan, np.nan, np.nan

    xy = tree_xyz[base_mask, :2]
    x_base, y_base = np.median(xy, axis=0)
    z_base = float(dtm.ground_z(np.array([[x_base, y_base]]))[0])
    return x_base, y_base, z_base


def _diameter_at_height(
    tree_xyz: np.ndarray, hag: np.ndarray, height_m: float, half_thickness_m: float,
    cfg: ForestMetricsConfig,
) -> dict:
    """Fit a single dedicated circle to a horizontal slice at `height_m`,
    shared by compute_dbh (1.3m) and the tree-validation check at
    tree_validation_height_m. Returns diameter_cm/cci/n_points/x/y, with
    diameter_cm NaN'd out (but cci kept, for diagnostics) when the fit's
    angular coverage is too poor to trust -- see compute_dbh for why."""
    half = half_thickness_m / 2
    slice_mask = np.abs(hag - height_m) <= half
    slice_xy = tree_xyz[slice_mask, :2]

    xc, yc, r, cci = fit_circle_robust(slice_xy, cfg)
    diameter_cm = 2 * r * 100 if np.isfinite(r) else np.nan
    if np.isfinite(diameter_cm) and cci < cfg.min_cci_for_valid_dbh:
        diameter_cm = np.nan
    return {
        "diameter_cm": diameter_cm,
        "cci": cci,
        "n_points": int(slice_xy.shape[0]),
        "x": xc if np.isfinite(diameter_cm) else np.nan,
        "y": yc if np.isfinite(diameter_cm) else np.nan,
    }


def compute_dbh(tree_xyz: np.ndarray, hag: np.ndarray, cfg: ForestMetricsConfig) -> dict:
    # A circle fit to a slice with poor angular coverage (points bunched on
    # one side, e.g. an occluded or partially-scanned stem) is only weakly
    # constrained and can converge on a wildly wrong radius even though the
    # fit itself "succeeds" -- verified against real TLS data, where sub-0.3
    # CCI fits averaged 2-4x the diameter of well-covered ones. Below the
    # threshold, the value isn't usable even as an estimate.
    fit = _diameter_at_height(tree_xyz, hag, cfg.dbh_height_m, cfg.dbh_slice_thickness_m, cfg)
    return {
        "dbh_cm": fit["diameter_cm"],
        "dbh_cci": fit["cci"],
        "dbh_n_points": fit["n_points"],
        # Fitted circle center, kept only alongside a usable diameter -- used
        # to place the DBH ring in visualization.build_diameter_circle_points.
        "dbh_x": fit["x"],
        "dbh_y": fit["y"],
    }


def compute_height(hag: np.ndarray, cfg: ForestMetricsConfig) -> float:
    if hag.shape[0] == 0:
        return np.nan
    return float(np.nanpercentile(hag, cfg.tree_height_percentile))


def compute_crown_metrics(tree_xyz: np.ndarray, hag: np.ndarray, tree_height: float, cfg: ForestMetricsConfig) -> dict:
    empty = {
        "crown_base_height_m": np.nan,
        "live_crown_ratio": np.nan,
        "crown_diameter_m": np.nan,
        "crown_area_m2": np.nan,
        "crown_volume_m3": np.nan,
        "crown_mean_x": np.nan,
        "crown_mean_y": np.nan,
        "crown_top_x": np.nan,
        "crown_top_y": np.nan,
        "crown_top_z": np.nan,
    }
    if tree_xyz.shape[0] == 0 or not np.isfinite(tree_height) or tree_height <= 0:
        return empty

    top_idx = np.nanargmax(hag)
    empty["crown_top_x"] = float(tree_xyz[top_idx, 0])
    empty["crown_top_y"] = float(tree_xyz[top_idx, 1])
    empty["crown_top_z"] = float(tree_xyz[top_idx, 2])

    bin_edges = np.arange(0, tree_height + cfg.crown_height_bin_m, cfg.crown_height_bin_m)
    if bin_edges.shape[0] < 2:
        return empty
    counts, _ = np.histogram(hag, bins=bin_edges)
    if counts.max() == 0:
        return empty

    threshold = cfg.crown_density_fraction * counts.max()
    crown_base_bin = None
    run = 0
    for i in range(len(counts) - 1, -1, -1):
        if counts[i] >= threshold:
            run += 1
            if run >= cfg.crown_min_consecutive_bins:
                crown_base_bin = i
        else:
            run = 0
    if crown_base_bin is None:
        # No stretch of bins met the consecutive-bin requirement; treat the
        # single highest-density bin as the crown base instead of giving up.
        crown_base_bin = int(np.argmax(counts))

    crown_base_height = bin_edges[crown_base_bin]
    empty["crown_base_height_m"] = float(crown_base_height)
    empty["live_crown_ratio"] = float((tree_height - crown_base_height) / tree_height)

    crown_mask = hag >= crown_base_height
    crown_pts = tree_xyz[crown_mask]
    crown_hag = hag[crown_mask]
    if crown_pts.shape[0] < 3:
        return empty

    empty["crown_mean_x"] = float(np.mean(crown_pts[:, 0]))
    empty["crown_mean_y"] = float(np.mean(crown_pts[:, 1]))

    try:
        hull = ConvexHull(crown_pts[:, :2])
        area = hull.volume
        empty["crown_area_m2"] = float(area)
        empty["crown_diameter_m"] = float(2 * np.sqrt(area / np.pi))
    except Exception:
        pass

    voxel = cfg.crown_voxel_size_m
    voxel_idx = np.floor(crown_pts / voxel).astype(np.int64)
    n_occupied = len(np.unique(voxel_idx, axis=0))
    empty["crown_volume_m3"] = float(n_occupied * voxel ** 3)

    return empty


def compute_taper(tree_xyz: np.ndarray, hag: np.ndarray, max_height_m: float, cfg: ForestMetricsConfig) -> pd.DataFrame:
    """Samples diameter from taper_height_min_m up to max_height_m, which the
    caller should pass as the commercial/merchantable trunk height (e.g.
    crown_base_height_m), not the tree's full height -- crown diameters
    aren't meaningful trunk taper and shouldn't be measured or reported."""
    columns = ["height_m", "diameter_cm", "cci", "n_points", "center_x", "center_y", "tilt_outlier_prob"]
    if not np.isfinite(max_height_m) or max_height_m <= cfg.taper_height_min_m:
        return pd.DataFrame(columns=columns)

    heights = np.arange(cfg.taper_height_min_m, max_height_m, cfg.taper_height_increment_m)
    rows = []
    half = cfg.taper_slice_thickness_m / 2
    for h in heights:
        slice_mask = np.abs(hag - h) <= half
        slice_xy = tree_xyz[slice_mask, :2]
        n_points = int(slice_xy.shape[0])
        if n_points >= cfg.min_points_for_taper_slice:
            xc, yc, r, cci = fit_circle_robust(slice_xy, cfg)
            diameter_cm = 2 * r * 100 if np.isfinite(r) else np.nan
            # Same low-coverage failure mode as compute_dbh: keep cci for
            # diagnostics but don't feed an unreliable diameter into volume.
            if np.isfinite(diameter_cm) and cci < cfg.min_cci_for_valid_dbh:
                diameter_cm = np.nan
            if not np.isfinite(diameter_cm):
                xc, yc = np.nan, np.nan
        else:
            diameter_cm, cci, xc, yc = np.nan, np.nan, np.nan, np.nan
        rows.append((h, diameter_cm, cci, n_points, xc, yc, 0.0))

    taper_df = pd.DataFrame(rows, columns=columns)

    # tilt_detection (ported from dendromatics): a section whose fitted
    # center is off-axis relative to the tree's other sections is a
    # different failure mode than poor radial coverage -- the fit can look
    # numerically fine (good CCI) while being centered on e.g. a branch
    # cluster rather than the stem. Needs every section's center at once,
    # so it runs as a second pass after the per-height loop above.
    radius_cm = taper_df["diameter_cm"].to_numpy(dtype=float) / 2
    radius_cm = np.nan_to_num(radius_cm, nan=0.0)
    tilt_prob = tilt_outlier_prob(
        taper_df["center_x"].to_numpy(dtype=float),
        taper_df["center_y"].to_numpy(dtype=float),
        radius_cm,
        taper_df["height_m"].to_numpy(dtype=float),
    )
    taper_df["tilt_outlier_prob"] = tilt_prob

    tilt_flagged = tilt_prob > cfg.tilt_outlier_threshold
    taper_df.loc[tilt_flagged, ["diameter_cm", "center_x", "center_y"]] = np.nan

    return taper_df


def apply_monotonic_correction(taper_df: pd.DataFrame, cfg: ForestMetricsConfig) -> pd.DataFrame:
    """Adds a diameter_corrected_cm column. Walking the taper curve bottom
    to top, a sample is flagged as physiologically implausible if its
    diameter exceeds the *smaller* of the last two accepted (not flagged)
    diameters below it -- a tree's diameter should not increase with
    height. Using the smaller of the two (rather than their average) is
    what actually guarantees the corrected curve never increases: an
    accepted value is by construction <= the accepted value right before
    it, so the accepted sequence is non-increasing by induction; the
    average alone doesn't have that property and can let the curve creep
    upward a little at a time. Flagged samples are replaced by linear
    interpolation between
    the nearest accepted samples below and above (extrapolated flat at
    either end if there's no accepted sample on that side), never by
    dropping them or folding them into a global fit.

    This is deliberately local: a bad run of samples only ever affects
    itself, never the clean samples around it. An earlier version of this
    function used a single weighted isotonic regression across the whole
    curve instead -- verified against real data that this can go wrong
    badly: a handful of high-height samples with bad measurements (e.g. an
    overlapping neighbour's crown) dragged the corrected diameter for an
    *entire* tree, including untouched clean low samples, to a single flat
    number nowhere near any of the raw measurements.
    """
    taper_df = taper_df.copy()
    taper_df["diameter_corrected_cm"] = np.nan

    valid = taper_df.dropna(subset=["diameter_cm"]).sort_values("height_m")
    if valid.empty:
        return taper_df

    heights = valid["height_m"].to_numpy(dtype=float)
    diameters = valid["diameter_cm"].to_numpy(dtype=float)

    accepted_heights = []
    accepted_diameters = []
    flagged = np.zeros(len(heights), dtype=bool)

    for i in range(len(heights)):
        if len(accepted_diameters) >= 2:
            reference = min(accepted_diameters[-1], accepted_diameters[-2])
            if diameters[i] > reference:
                flagged[i] = True
        if not flagged[i]:
            accepted_heights.append(heights[i])
            accepted_diameters.append(diameters[i])

    corrected = diameters.copy()
    if flagged.any() and accepted_diameters:
        # np.interp extrapolates flat beyond the accepted range, i.e. holds
        # the nearest accepted diameter at either end of the curve when
        # there's no accepted sample on that side to interpolate between.
        corrected[flagged] = np.interp(heights[flagged], accepted_heights, accepted_diameters)

    taper_df.loc[valid.index, "diameter_corrected_cm"] = corrected
    return taper_df


def measure_tree(tree_id: int, tree_df: pd.DataFrame, dtm: DTM, cfg: ForestMetricsConfig) -> Tuple[dict, pd.DataFrame]:
    """Top-level per-tree measurement pipeline. Never raises for data-quality
    problems -- always returns a row (possibly mostly NaN) plus a taper
    DataFrame. row["is_valid_tree"] is False for segmented instances that
    don't pass the tree-vs-shrub/plot-edge-fragment check (too short to
    reach tree_validation_height_m, or lacking a consistent, well-covered
    circular cross-section at several heights below it) -- forest_metrics.py
    drops those entirely rather than including them in any output."""
    flags = []

    tree_xyz = tree_df[["X", "Y", "Z"]].values
    n_points = int(tree_xyz.shape[0])
    if n_points < cfg.min_points_per_tree:
        flags.append("too_few_points")

    hag = dtm.height_above_ground(tree_xyz)

    x_base, y_base, z_base = compute_tree_base(tree_xyz, hag, dtm, cfg)
    dbh = compute_dbh(tree_xyz, hag, cfg)
    if not np.isfinite(dbh["dbh_cm"]):
        # compute_dbh nulls dbh_cm both when no circle could be fit at all
        # (dbh_cci also NaN) and when a circle fit but coverage was too low
        # to trust (dbh_cci is a real, sub-threshold value) -- distinguish
        # the two so the flag says which one happened.
        if np.isfinite(dbh["dbh_cci"]) and dbh["dbh_cci"] < cfg.min_cci_for_valid_dbh:
            flags.append("low_cci")
        else:
            flags.append("no_dbh_slice")

    height_m = compute_height(hag, cfg)
    crown = compute_crown_metrics(tree_xyz, hag, height_m, cfg)

    # Diameters are only meaningful -- and only reported -- below the crown.
    # crown_base_height_m can fall slightly above tree_height (its bin edges
    # are built past tree_height by up to one crown_height_bin_m), so clamp.
    # It can also come back at or near 0: the density-profile heuristic
    # assumes a sparse trunk vs. a dense crown, which doesn't hold for a
    # densely-scanned TLS tree where point density stays roughly uniform
    # top to bottom -- verified against real data, ~1 in 4 trees. A tree
    # that reached this point has a real DBH ring at breast height, so any
    # "crown" reported at or below it is a degenerate reading, not a real
    # trunk-less tree; fall back to the full height in that case rather
    # than reporting an empty taper curve.
    commercial_height_m = crown["crown_base_height_m"]
    if np.isfinite(commercial_height_m) and commercial_height_m > cfg.dbh_height_m:
        commercial_height_m = min(commercial_height_m, height_m)
    else:
        commercial_height_m = height_m

    # Sample taper up to whichever is taller -- the commercial trunk height
    # (needed for reporting) or tree_validation_height_m (needed for the
    # tree-vs-not-a-tree check below) -- then split the two uses apart.
    # Correction runs on the *full* sampled range before truncating for
    # reporting: a trusted slice between commercial_height_m and
    # tree_validation_height_m (part of why the tree passed validation at
    # all) is real evidence the corrected curve should use, even though
    # that row itself won't appear in the reported taper.
    sampling_height_m = min(max(commercial_height_m, cfg.tree_validation_height_m), height_m)
    full_taper_df = compute_taper(tree_xyz, hag, sampling_height_m, cfg)
    full_taper_df = apply_monotonic_correction(full_taper_df, cfg)

    taper_df = full_taper_df[full_taper_df["height_m"] <= commercial_height_m].reset_index(drop=True)
    if taper_df.empty or taper_df["diameter_cm"].isna().all():
        flags.append("no_taper_data")
    taper_df.insert(0, "tree_id", tree_id)

    # Tree-vs-not-a-tree: must actually reach tree_validation_height_m (the
    # shrub/undergrowth filter) AND show a consistent, well-covered
    # circular cross-section at several heights between the ground and
    # there, not just one -- a single high-CCI slice can happen on a branch
    # cluster, and a single low-CCI slice can happen on a real trunk (see
    # config.py for the real-data case that motivated counting several
    # slices instead of trusting one). DBH counts as one of the slices.
    validation_slices = full_taper_df[full_taper_df["height_m"] <= cfg.tree_validation_height_m]
    high_cci_count = int((validation_slices["cci"] >= cfg.tree_validation_cci_threshold).sum())
    if np.isfinite(dbh["dbh_cci"]) and dbh["dbh_cci"] >= cfg.tree_validation_cci_threshold:
        high_cci_count += 1

    is_valid_tree = bool(
        np.isfinite(dbh["dbh_cm"])
        and dbh["dbh_cm"] >= cfg.min_valid_dbh_cm
        and height_m >= cfg.tree_validation_height_m
        and high_cci_count >= cfg.min_high_cci_slices
    )
    if not is_valid_tree:
        flags.append("rejected_not_a_tree")

    row = {
        "tree_id": tree_id,
        "n_points": n_points,
        "x_base": x_base,
        "y_base": y_base,
        "z_base": z_base,
        "height_m": height_m,
        "is_valid_tree": is_valid_tree,
        "quality_flags": ";".join(flags),
    }
    row.update(dbh)
    row.update(crown)

    return row, taper_df
