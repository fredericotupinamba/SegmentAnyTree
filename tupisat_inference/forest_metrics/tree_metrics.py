from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.measure import CircleModel, ransac
from sklearn.isotonic import IsotonicRegression

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import DTM

NAN_CIRCLE = (np.nan, np.nan, np.nan, np.nan)


def fit_circle_ransac(points_xy: np.ndarray, cfg: ForestMetricsConfig, max_trials: int) -> Tuple[float, float, float, float]:
    """RANSAC circle fit on a horizontal slice of points.
    Returns (xc, yc, radius_m, cci); all NaN if there aren't enough points
    or the fit fails."""
    n = points_xy.shape[0]
    if n < cfg.min_points_for_circle_fit:
        return NAN_CIRCLE

    # min_samples grows with slice density in principle, but a dense TLS
    # slice can have hundreds of points -- capping it keeps each RANSAC
    # trial's per-sample circle fit cheap without hurting robustness (a
    # circle only needs 3 points; a few dozen already gives a solid,
    # outlier-resistant fit).
    min_samples = min(max(3, int(0.3 * n)), cfg.circle_ransac_min_samples_cap)

    # points_xy arrives in real-world UTM-scale coordinates (6-7 digit
    # values). skimage 0.16.2's CircleModel (pinned in production) solves
    # the fit via a plain least-squares linear system that is numerically
    # unstable at that magnitude -- verified against real data: fitting a
    # perfectly clean, dense 2000-point stem ring in raw coordinates
    # produced a "circle" with a ~2,300,000 m radius, while the same points
    # shifted to be centered on zero fit correctly (radius ~0.16 m).
    # Newer skimage tolerates raw coordinates fine, which is why this only
    # showed up running against the actual pinned Docker environment, not
    # local testing with an unpinned newer skimage. Center before fitting,
    # then shift the result back.
    centroid = points_xy.mean(axis=0)
    centered_xy = points_xy - centroid

    try:
        model, inliers = ransac(
            centered_xy,
            CircleModel,
            min_samples=min_samples,
            residual_threshold=cfg.circle_ransac_residual_threshold_m,
            max_trials=max_trials,
        )
    except (ValueError, np.linalg.LinAlgError):
        return NAN_CIRCLE

    # Older skimage returns None on a failed fit; newer skimage returns a
    # FailedEstimation object that is falsy but raises on attribute access
    # (e.g. "not enough significant data points", "no inliers found") --
    # bool() is the version-agnostic way to detect either case.
    if model is None or not model:
        return NAN_CIRCLE

    xc, yc, r = model.params
    if not np.isfinite(r) or r <= 0 or 2 * r * 100 > cfg.max_plausible_diameter_cm:
        return NAN_CIRCLE
    xc, yc = xc + centroid[0], yc + centroid[1]

    cci = circumferential_completeness_index((xc, yc), r, points_xy)
    return xc, yc, r, cci


def circumferential_completeness_index(
    center_xy: Tuple[float, float], radius: float, points_xy: np.ndarray, sector_deg: float = 4.5
) -> float:
    """Fraction of angular sectors around the fitted circle that contain at
    least one point within [0.8r, 1.2r] of the center."""
    if radius <= 0 or points_xy.shape[0] == 0:
        return 0.0

    xc, yc = center_xy
    dx = points_xy[:, 0] - xc
    dy = points_xy[:, 1] - yc
    dist = np.hypot(dx, dy)
    ring_mask = (dist >= 0.8 * radius) & (dist <= 1.2 * radius)
    if not np.any(ring_mask):
        return 0.0

    angles_deg = np.degrees(np.arctan2(dy[ring_mask], dx[ring_mask])) % 360
    n_sectors = int(np.ceil(360 / sector_deg))
    sector_idx = np.floor(angles_deg / sector_deg).astype(int)
    return len(np.unique(sector_idx)) / n_sectors


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
    cfg: ForestMetricsConfig, max_trials: int,
) -> dict:
    """Fit a single dedicated circle to a horizontal slice at `height_m`,
    shared by compute_dbh (1.3m) and the tree-validation check at
    tree_validation_height_m. Returns diameter_cm/cci/n_points/x/y, with
    diameter_cm NaN'd out (but cci kept, for diagnostics) when the fit's
    angular coverage is too poor to trust -- see compute_dbh for why."""
    half = half_thickness_m / 2
    slice_mask = np.abs(hag - height_m) <= half
    slice_xy = tree_xyz[slice_mask, :2]

    xc, yc, r, cci = fit_circle_ransac(slice_xy, cfg, max_trials=max_trials)
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
    fit = _diameter_at_height(tree_xyz, hag, cfg.dbh_height_m, cfg.dbh_slice_thickness_m, cfg, cfg.dbh_ransac_max_trials)
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
    columns = ["height_m", "diameter_cm", "cci", "n_points", "center_x", "center_y"]
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
            xc, yc, r, cci = fit_circle_ransac(slice_xy, cfg, max_trials=cfg.taper_ransac_max_trials)
            diameter_cm = 2 * r * 100 if np.isfinite(r) else np.nan
            # Same low-coverage failure mode as compute_dbh: keep cci for
            # diagnostics but don't feed an unreliable diameter into volume.
            if np.isfinite(diameter_cm) and cci < cfg.min_cci_for_valid_dbh:
                diameter_cm = np.nan
            if not np.isfinite(diameter_cm):
                xc, yc = np.nan, np.nan
        else:
            diameter_cm, cci, xc, yc = np.nan, np.nan, np.nan, np.nan
        rows.append((h, diameter_cm, cci, n_points, xc, yc))

    return pd.DataFrame(rows, columns=columns)


def apply_monotonic_correction(
    taper_df: pd.DataFrame, dbh_cm: float, dbh_height_m: float, dbh_n_points: int, dbh_cci: float,
    cfg: ForestMetricsConfig,
) -> pd.DataFrame:
    """Adds a diameter_corrected_cm column. A tree's diameter should never
    increase with height (ignoring base flare, which this ignores as a
    minor simplification, same as most taper models) -- but a branch, a
    neighbouring tree's crown, or any other slice with poor angular
    coverage can produce a diameter that's not just noisy but wildly wrong.

    Only slices with cci >= tree_validation_cci_threshold (the same "is
    this really a full, well-covered cross-section" bar used for the
    tree-validity check) are used as *input* to the fit -- a low-CCI slice
    is not weak evidence to be down-weighted, it is not evidence at all,
    and including it (even weighted by n_points) can drag the *entire*
    non-increasing curve toward a bad number, since a garbage slice isn't
    guaranteed to carry fewer points than a clean one. Verified against
    real data: a tree with 9 consecutive non-NaN-but-low-CCI slices (0.3-
    0.68), several with more points than the clean slices below them,
    pulled the corrected diameter for the entire tree -- including
    untouched cci=1.0 samples -- to a single flat, physically implausible
    value.

    A high CCI alone still isn't sufficient, though: it means the circle
    fit is well-supported by points on its circumference, not that those
    points are actually this tree's own trunk rather than an overlapping
    neighbour's crown or stem. Slices above breast height are additionally
    dropped from the fit if their diameter exceeds DBH by more than
    max_taper_over_dbh_ratio -- verified against real data where several
    such slices individually passed the CCI bar but, left in, forced the
    whole non-increasing curve up to match them (see config.py for the
    exact numbers).

    Every originally-measured height still gets a corrected value (via the
    fitted curve's prediction/extrapolation), including the untrusted
    ones -- they just don't get a vote in shaping that curve.
    """
    taper_df = taper_df.copy()
    taper_df["diameter_corrected_cm"] = np.nan

    valid = taper_df.dropna(subset=["diameter_cm"])
    if valid.empty:
        return taper_df

    trusted = valid[valid["cci"] >= cfg.tree_validation_cci_threshold]
    if np.isfinite(dbh_cm):
        implausible = (trusted["height_m"] > dbh_height_m) & (
            trusted["diameter_cm"] > dbh_cm * cfg.max_taper_over_dbh_ratio
        )
        trusted = trusted[~implausible]

    heights = list(trusted["height_m"].values)
    diameters = list(trusted["diameter_cm"].values)
    weights = list(trusted["n_points"].values.astype(float))

    if np.isfinite(dbh_cm) and (not np.isfinite(dbh_cci) or dbh_cci >= cfg.tree_validation_cci_threshold):
        heights.append(dbh_height_m)
        diameters.append(dbh_cm)
        weights.append(float(max(dbh_n_points, 1)))

    if not heights:
        # No trustworthy slice anywhere on this tree to correct against --
        # leaving diameter_corrected_cm as NaN here (rather than copying
        # the raw, untrusted values through) matters: those raw values are
        # exactly the ones this function exists to not trust, and passing
        # them through unmodified can reintroduce the non-increasing
        # violations the rest of the pipeline (volume, visualization)
        # assumes are already resolved.
        return taper_df

    heights = np.asarray(heights)
    diameters = np.asarray(diameters)
    weights = np.asarray(weights)
    order = np.argsort(heights)

    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(heights[order], diameters[order], sample_weight=weights[order])

    taper_df.loc[valid.index, "diameter_corrected_cm"] = model.predict(valid["height_m"].values)
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
    full_taper_df = apply_monotonic_correction(
        full_taper_df, dbh["dbh_cm"], cfg.dbh_height_m, dbh["dbh_n_points"], dbh["dbh_cci"], cfg
    )

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
