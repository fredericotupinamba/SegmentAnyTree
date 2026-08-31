#!/usr/bin/env python3
"""Extract per-height-bin features for fitting the crown-base-height model.

Replaces the leaf-contaminated feature set this module's first version
produced (kept as _legacy_extract_cbh_features.py). That version binned
*all* tree points together, so a bin's footprint area and median intensity
mixed bark and foliage -- the very two classes the crown base separates.
With a PointsToWood wood/leaf label per point (`prediction`: 0 = leaf,
1 = wood) the same bins can be described by what is actually diagnostic:
how much of the bin is wood, and how wide that wood spreads.

Input is a `*_pwood.laz` -- a SAT merged cloud (PredSemantic/PredInstance/
intensity) that has been through PointsToWood, so it also carries
`prediction` and `pwood`. Only trees named in the label CSV are processed.

Bin geometry (0.25 m bins, 0.10 m footprint cells) and the trunk-baseline
convention (bins at or below dbh_height_m) are taken from
ForestMetricsConfig, so features here mean the same thing they will mean
inside compute_crown_metrics once the fitted model is wired in.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import laspy
import numpy as np
import pandas as pd

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import build_dtm

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABELS = os.path.join(HERE, "cbh_tree_labels.csv")


def occupied_area_m2(bin_idx, xy, n_bins, cell_m):
    """Footprint area per bin as occupied-cell count x cell area -- robust
    to a single stray point the way a convex hull is not. Vectorized the
    same way compute_crown_metrics does it: one np.unique over
    (bin, cell_x, cell_y) triples rather than a per-bin np.unique(axis=0),
    which dominates runtime on multi-million-point trees."""
    if bin_idx.shape[0] == 0:
        return np.zeros(n_bins, dtype=float)
    cell_x = np.floor(xy[:, 0] / cell_m).astype(np.int64)
    cell_y = np.floor(xy[:, 1] / cell_m).astype(np.int64)
    triples = np.unique(np.column_stack([bin_idx, cell_x, cell_y]), axis=0)
    return np.bincount(triples[:, 0], minlength=n_bins).astype(float) * (cell_m ** 2)


def spread_per_bin(bin_idx, xy, n_bins, percentile):
    """Horizontal spread of the given points within each bin: the
    `percentile`-th distance from that bin's own median (x, y).

    Measuring against each bin's own centre rather than a single stem axis
    makes this lean-invariant -- a tilted trunk keeps a small spread all
    the way up, where a fixed-axis radius would grow with lean and fake a
    branch. On a clean bole the spread is ~half the stem diameter; the
    first real branch makes it jump."""
    out = np.full(n_bins, np.nan)
    if bin_idx.shape[0] == 0:
        return out
    order = np.argsort(bin_idx, kind="stable")
    sorted_bins = bin_idx[order]
    sorted_xy = xy[order]
    starts = np.searchsorted(sorted_bins, np.arange(n_bins), side="left")
    ends = np.searchsorted(sorted_bins, np.arange(n_bins), side="right")
    for i in range(n_bins):
        if ends[i] - starts[i] < 2:
            continue
        pts = sorted_xy[starts[i]:ends[i]]
        centre = np.median(pts, axis=0)
        out[i] = float(np.percentile(np.linalg.norm(pts - centre, axis=1), percentile))
    return out


def median_per_bin(bin_idx, values, n_bins):
    out = np.full(n_bins, np.nan)
    if bin_idx.shape[0] == 0:
        return out
    order = np.argsort(bin_idx, kind="stable")
    sorted_bins = bin_idx[order]
    sorted_vals = values[order]
    starts = np.searchsorted(sorted_bins, np.arange(n_bins), side="left")
    ends = np.searchsorted(sorted_bins, np.arange(n_bins), side="right")
    for i in range(n_bins):
        if ends[i] > starts[i]:
            out[i] = float(np.median(sorted_vals[starts[i]:ends[i]]))
    return out


def baseline(values, bin_mid, counts, cfg):
    """Per-tree trunk baseline for a bin-indexed feature: its median over
    the bins at or below breast height that hold enough points to trust.
    Falls back to all trunk-region bins when none clear the count gate
    (only happens on very sparsely scanned trees)."""
    eligible = (bin_mid <= cfg.dbh_height_m) & (counts >= cfg.crown_baseline_min_points_per_bin)
    if not np.any(eligible):
        eligible = bin_mid <= cfg.dbh_height_m
    if not np.any(eligible):
        return np.nan
    return float(np.nanmedian(values[eligible]))


def features_for_plot(plot_name, laz_path, label_rows, cfg, spread_percentile):
    print(f"=== {plot_name}: reading {os.path.basename(laz_path)} ===", flush=True)
    las = laspy.read(laz_path)
    xyz_all = np.column_stack([
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
    ])
    intensity_all = np.asarray(las.intensity, dtype=np.float64)
    pred_sem = np.asarray(las.PredSemantic)
    pred_inst = np.asarray(las.PredInstance)

    dims = set(las.point_format.dimension_names)
    if "prediction" not in dims:
        raise ValueError(
            f"{laz_path} has no 'prediction' dimension -- expected a PointsToWood "
            f"output (*_pwood.laz). Available: {sorted(dims)}"
        )
    # PointsToWood writes prediction as float64 (0.0 / 1.0), not a flag.
    is_wood_all = np.asarray(las.prediction, dtype=np.float64) >= 0.5
    print(f"{plot_name}: {len(xyz_all):,} points, {int(is_wood_all.sum()):,} wood", flush=True)

    print(f"{plot_name}: building dtm...", flush=True)
    dtm, warnings = build_dtm(xyz_all[pred_sem == 0], xyz_all, cfg)
    for w in warnings:
        print(f"{plot_name}: WARNING {w}", flush=True)

    tree_mask_all = (pred_sem == 1) & (pred_inst > 0)
    rows = []

    for label in label_rows:
        tid = int(label["tree_id"])
        mask = tree_mask_all & (pred_inst == tid)
        n = int(mask.sum())
        if n == 0:
            print(f"{plot_name} tree {tid}: no points, skipping", flush=True)
            continue

        txyz = xyz_all[mask]
        t_int = intensity_all[mask]
        t_wood = is_wood_all[mask]
        hag = dtm.height_above_ground(txyz)
        tree_height = float(np.nanpercentile(hag, cfg.tree_height_percentile))

        bin_edges = np.arange(0, tree_height + cfg.crown_height_bin_m, cfg.crown_height_bin_m)
        if bin_edges.shape[0] < 2:
            continue
        n_bins = bin_edges.shape[0] - 1
        bin_idx = np.clip(np.digitize(hag, bin_edges) - 1, 0, n_bins - 1)

        counts = np.bincount(bin_idx, minlength=n_bins)
        wood_idx = bin_idx[t_wood]
        counts_wood = np.bincount(wood_idx, minlength=n_bins)

        area_all = occupied_area_m2(bin_idx, txyz[:, :2], n_bins, cfg.crown_footprint_cell_m)
        area_wood = occupied_area_m2(wood_idx, txyz[t_wood][:, :2], n_bins, cfg.crown_footprint_cell_m)
        spread_wood = spread_per_bin(wood_idx, txyz[t_wood][:, :2], n_bins, spread_percentile)
        int_med_wood = median_per_bin(wood_idx, t_int[t_wood], n_bins)
        int_med_all = median_per_bin(bin_idx, t_int, n_bins)

        bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Normalise every scale-carrying feature by the tree's own trunk so
        # the fitted model does not have to learn one threshold per DBH
        # class -- these 58 trees span 24-45 cm DBH.
        area_wood0 = baseline(area_wood, bin_mid, counts_wood, cfg)
        spread_wood0 = baseline(spread_wood, bin_mid, counts_wood, cfg)
        int_wood0 = baseline(int_med_wood, bin_mid, counts_wood, cfg)

        with np.errstate(divide="ignore", invalid="ignore"):
            wood_frac = np.where(counts > 0, counts_wood / np.maximum(counts, 1), np.nan)

        for i in range(n_bins):
            if counts[i] == 0:
                continue
            rows.append({
                "plot": plot_name,
                "tree_id": tid,
                "bin_low": float(bin_edges[i]),
                "bin_high": float(bin_edges[i + 1]),
                "hag_mid": float(bin_mid[i]),
                "tree_height_m": tree_height,
                "CBH_m": float(label["CBH_m"]),
                "n_points": int(counts[i]),
                "n_wood": int(counts_wood[i]),
                "n_leaf": int(counts[i] - counts_wood[i]),
                "wood_frac": float(wood_frac[i]),
                "leaf_frac": float(1.0 - wood_frac[i]),
                "wood_area_m2": float(area_wood[i]),
                "total_area_m2": float(area_all[i]),
                "wood_spread_m": float(spread_wood[i]),
                "intensity_median_wood": float(int_med_wood[i]),
                "intensity_median_all": float(int_med_all[i]),
                # Trunk-normalised forms -- the ones the model should use.
                "wood_area_ratio": float(area_wood[i] / area_wood0) if area_wood0 else np.nan,
                "wood_spread_ratio": float(spread_wood[i] / spread_wood0) if spread_wood0 else np.nan,
                "intensity_wood_ratio": float(int_med_wood[i] / int_wood0) if int_wood0 else np.nan,
            })

        print(f"{plot_name} tree {tid}: n={n:,} wood={int(t_wood.sum()):,} "
              f"H={tree_height:.2f} CBH={float(label['CBH_m']):.2f} -> {n_bins} bins", flush=True)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pwood", action="append", required=True, metavar="PLOT=PATH",
                         help="Plot name and its *_pwood.laz, e.g. P01=data/05-PWOOD/x_pwood.laz. Repeatable.")
    parser.add_argument("--labels", default=DEFAULT_LABELS,
                         help="Manual crown-base-height labels CSV (default: %(default)s).")
    parser.add_argument("--output", "-o", default=os.path.join(HERE, "cbh_bin_features_wood.csv"))
    parser.add_argument("--spread-percentile", type=float, default=95.0,
                         help="Percentile used for wood_spread_m (default: %(default)s).")
    args = parser.parse_args()

    cfg = ForestMetricsConfig()
    labels_df = pd.read_csv(args.labels)
    labels_df = labels_df[labels_df["CBH_m"].notna() & (labels_df["CBH_m"] < 100)]

    all_rows = []
    for spec in args.pwood:
        if "=" not in spec:
            raise SystemExit(f"--pwood expects PLOT=PATH, got: {spec}")
        plot_name, laz_path = spec.split("=", 1)
        plot_labels = labels_df[labels_df["plot"] == plot_name].to_dict("records")
        if not plot_labels:
            print(f"WARNING: no labels for plot {plot_name}, skipping", flush=True)
            continue
        all_rows.extend(features_for_plot(plot_name, laz_path, plot_labels, cfg, args.spread_percentile))

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output, index=False)
    n_trees = out_df.groupby(["plot", "tree_id"]).ngroups if not out_df.empty else 0
    print(f"\nWrote {len(out_df):,} bin rows for {n_trees} tree(s) -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
