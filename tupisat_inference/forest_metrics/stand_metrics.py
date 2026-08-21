import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import DTM


def compute_plot_area_ha(dtm: DTM, cfg: ForestMetricsConfig) -> float:
    if cfg.plot_area_mode == "fixed_radius":
        if cfg.fixed_plot_radius_m is None:
            raise ValueError("plot_area_mode is 'fixed_radius' but fixed_plot_radius_m is not set.")
        return np.pi * cfg.fixed_plot_radius_m ** 2 / 10000
    return dtm.convex_hull_area_m2() / 10000


def compute_canopy_cover_fraction(tree_pts_xyz: np.ndarray, hag: np.ndarray, dtm: DTM, cfg: ForestMetricsConfig) -> float:
    if tree_pts_xyz.shape[0] == 0:
        return np.nan

    canopy_mask = hag >= cfg.canopy_min_height_above_ground_m
    canopy_xy = tree_pts_xyz[canopy_mask, :2]

    (xmin, ymin), (xmax, ymax) = dtm.xy_bounds
    x_edges = np.arange(xmin, xmax + cfg.canopy_grid_resolution_m, cfg.canopy_grid_resolution_m)
    y_edges = np.arange(ymin, ymax + cfg.canopy_grid_resolution_m, cfg.canopy_grid_resolution_m)
    if x_edges.shape[0] == 0 or y_edges.shape[0] == 0:
        return np.nan

    canopy_tree = cKDTree(canopy_xy) if canopy_xy.shape[0] > 0 else None

    n_cells = 0
    n_canopy = 0
    for x in x_edges:
        for y in y_edges:
            n_cells += 1
            if canopy_tree is None:
                continue
            idx = canopy_tree.query_ball_point([x, y], r=cfg.canopy_grid_resolution_m)
            if len(idx) > cfg.canopy_min_points_per_cell:
                n_canopy += 1

    return n_canopy / n_cells if n_cells > 0 else np.nan


def compute_spacing_indices(tree_metrics_df: pd.DataFrame, plot_area_ha: float) -> dict:
    result = {"mean_nn_distance_m": np.nan, "clark_evans_r": np.nan}

    bases = tree_metrics_df[["x_base", "y_base"]].dropna().values
    if bases.shape[0] < 2 or not np.isfinite(plot_area_ha) or plot_area_ha <= 0:
        return result

    tree = cKDTree(bases)
    dist, _ = tree.query(bases, k=2)
    mean_nn_distance_m = float(dist[:, 1].mean())
    result["mean_nn_distance_m"] = mean_nn_distance_m

    density_per_m2 = bases.shape[0] / (plot_area_ha * 10000)
    if density_per_m2 > 0:
        expected_nn = 1 / (2 * np.sqrt(density_per_m2))
        result["clark_evans_r"] = mean_nn_distance_m / expected_nn

    return result


def compute_distribution_stats(series: pd.Series, bin_width: float, prefix: str) -> dict:
    result = {}
    valid = series.dropna()
    if valid.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
        }

    result[f"{prefix}_mean"] = float(valid.mean())
    result[f"{prefix}_std"] = float(valid.std())
    result[f"{prefix}_min"] = float(valid.min())
    result[f"{prefix}_max"] = float(valid.max())
    result[f"{prefix}_median"] = float(valid.median())

    lo = np.floor(valid.min() / bin_width) * bin_width
    hi = np.ceil(valid.max() / bin_width) * bin_width
    if hi <= lo:
        hi = lo + bin_width
    bins = np.arange(lo, hi + bin_width, bin_width)
    counts, edges = np.histogram(valid, bins=bins)
    for i, count in enumerate(counts):
        result[f"{prefix}_bin_{edges[i]:g}_{edges[i + 1]:g}_count"] = int(count)

    return result


def summarize_plot(tree_metrics_df: pd.DataFrame, dtm: DTM, tree_pts_df: pd.DataFrame, cfg: ForestMetricsConfig) -> dict:
    plot_area_ha = compute_plot_area_ha(dtm, cfg)
    n_trees = int(tree_metrics_df.shape[0])

    summary = {
        "n_trees": n_trees,
        "plot_area_ha": plot_area_ha,
        "trees_per_ha": (n_trees / plot_area_ha) if plot_area_ha > 0 else np.nan,
        "dtm_quality": "ok",
    }

    if n_trees == 0:
        summary["basal_area_m2_ha"] = np.nan
        summary["canopy_cover_fraction"] = np.nan
        summary.update(compute_distribution_stats(pd.Series(dtype=float), cfg.dbh_histogram_bin_cm, "dbh_cm"))
        summary.update(compute_spacing_indices(tree_metrics_df, plot_area_ha))
        return summary

    dbh_m = tree_metrics_df["dbh_cm"].dropna() / 100
    basal_area_m2 = float((np.pi / 4 * dbh_m ** 2).sum())
    summary["basal_area_m2_ha"] = (basal_area_m2 / plot_area_ha) if plot_area_ha > 0 else np.nan

    if tree_pts_df.shape[0] > 0:
        tree_pts_xyz = tree_pts_df[["X", "Y", "Z"]].values
        hag = dtm.height_above_ground(tree_pts_xyz)
        summary["canopy_cover_fraction"] = compute_canopy_cover_fraction(tree_pts_xyz, hag, dtm, cfg)
    else:
        summary["canopy_cover_fraction"] = np.nan

    summary.update(compute_distribution_stats(tree_metrics_df["dbh_cm"], cfg.dbh_histogram_bin_cm, "dbh_cm"))
    summary.update(compute_spacing_indices(tree_metrics_df, plot_area_ha))

    return summary
