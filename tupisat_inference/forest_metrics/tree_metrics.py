from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.measure import CircleModel, ransac

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


def compute_dbh(tree_xyz: np.ndarray, hag: np.ndarray, cfg: ForestMetricsConfig) -> dict:
    half = cfg.dbh_slice_thickness_m / 2
    slice_mask = np.abs(hag - cfg.dbh_height_m) <= half
    slice_xy = tree_xyz[slice_mask, :2]

    xc, yc, r, cci = fit_circle_ransac(slice_xy, cfg, max_trials=cfg.dbh_ransac_max_trials)
    dbh_cm = 2 * r * 100 if np.isfinite(r) else np.nan
    # A circle fit to a slice with poor angular coverage (points bunched on
    # one side, e.g. an occluded or partially-scanned stem) is only weakly
    # constrained and can converge on a wildly wrong radius even though the
    # fit itself "succeeds" -- verified against real TLS data, where sub-0.3
    # CCI fits averaged 2-4x the diameter of well-covered ones. Below the
    # threshold, the value isn't usable even as an estimate.
    if np.isfinite(dbh_cm) and cci < cfg.min_cci_for_valid_dbh:
        dbh_cm = np.nan
    return {
        "dbh_cm": dbh_cm,
        "dbh_cci": cci,
        "dbh_n_points": int(slice_xy.shape[0]),
        # Fitted circle center, kept only alongside a usable diameter -- used
        # to place the DBH ring in visualization.build_diameter_circle_points.
        "dbh_x": xc if np.isfinite(dbh_cm) else np.nan,
        "dbh_y": yc if np.isfinite(dbh_cm) else np.nan,
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


def compute_taper(tree_xyz: np.ndarray, hag: np.ndarray, tree_height: float, cfg: ForestMetricsConfig) -> pd.DataFrame:
    columns = ["height_m", "diameter_cm", "cci", "n_points", "center_x", "center_y"]
    if not np.isfinite(tree_height) or tree_height <= cfg.taper_height_min_m:
        return pd.DataFrame(columns=columns)

    heights = np.arange(cfg.taper_height_min_m, tree_height, cfg.taper_height_increment_m)
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


def measure_tree(tree_id: int, tree_df: pd.DataFrame, dtm: DTM, cfg: ForestMetricsConfig) -> Tuple[dict, pd.DataFrame]:
    """Top-level per-tree measurement pipeline. Never raises for data-quality
    problems -- always returns a row (possibly mostly NaN) plus a taper
    DataFrame."""
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
    taper_df = compute_taper(tree_xyz, hag, height_m, cfg)
    if taper_df.empty or taper_df["diameter_cm"].isna().all():
        flags.append("no_taper_data")
    taper_df = taper_df.copy()
    taper_df.insert(0, "tree_id", tree_id)

    row = {
        "tree_id": tree_id,
        "n_points": n_points,
        "x_base": x_base,
        "y_base": y_base,
        "z_base": z_base,
        "height_m": height_m,
        "quality_flags": ";".join(flags),
    }
    row.update(dbh)
    row.update(crown)

    return row, taper_df
