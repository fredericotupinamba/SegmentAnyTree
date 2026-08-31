"""HISTORICAL -- not runnable as-is.

This is the calibration that produced the intensity+area crown rule and
its published 1.26m figure, kept so that comparison can be audited. Its
paths point at a scratch directory that no longer exists, and its feature
set predates the wood/leaf labels. The current, runnable versions are
extract_cbh_features.py and calibrate_cbh.py beside it.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import laspy

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import build_dtm

BIN_M = 0.25
CELL_M = 0.10  # horizontal cell size for footprint-area proxy

PLOTS = {
    "P01": {
        "laz": "data/Teste_output/2025_04_28_HLS_P01_16m_out_SAT_output/2025_04_28_HLS_P01_16m_out.laz",
        "metrics": "data/Teste_output/2025_04_28_HLS_P01_16m_out_SAT_output/2025_04_28_HLS_P01_16m_out_tree_metrics.csv",
    },
    "P02": {
        "laz": "data/Teste_output/2025_04_28_HLS_P02_16m_out_SAT_output/2025_04_28_HLS_P02_16m_out.laz",
        "metrics": "data/Teste_output/2025_04_28_HLS_P02_16m_out_SAT_output/2025_04_28_HLS_P02_16m_out_tree_metrics.csv",
    },
}

OUT_BINS = "C:/Users/Fred/AppData/Local/Temp/claude/e--GITHUB-SegmentAnyTree/17ed6f96-d6d2-488a-946c-0426577ade70/scratchpad/cbh_bin_features.csv"
OUT_TREES = "C:/Users/Fred/AppData/Local/Temp/claude/e--GITHUB-SegmentAnyTree/17ed6f96-d6d2-488a-946c-0426577ade70/scratchpad/cbh_tree_labels.csv"

cfg = ForestMetricsConfig()

all_bin_rows = []
all_tree_rows = []

for plot_name, paths in PLOTS.items():
    print(f"=== {plot_name}: loading LAZ ===", flush=True)
    las = laspy.read(paths["laz"])
    X = np.asarray(las.x, dtype=np.float64)
    Y = np.asarray(las.y, dtype=np.float64)
    Z = np.asarray(las.z, dtype=np.float64)
    intensity = np.asarray(las.intensity, dtype=np.float64)
    pred_sem = np.asarray(las.PredSemantic)
    pred_inst = np.asarray(las.PredInstance)
    print(f"{plot_name}: {len(X)} points loaded", flush=True)

    xyz_all = np.column_stack([X, Y, Z])
    non_tree_xyz = xyz_all[pred_sem == 0]
    print(f"{plot_name}: building dtm...", flush=True)
    dtm, warnings = build_dtm(non_tree_xyz, xyz_all, cfg)
    print(f"{plot_name}: dtm built", flush=True)

    labels_df = pd.read_csv(paths["metrics"])
    labels_df = labels_df[labels_df["CBH_m"].notna()]

    tree_mask_all = (pred_sem == 1) & (pred_inst > 0)

    for _, row in labels_df.iterrows():
        tid = int(row["tree_id"])
        cbh_label = float(row["CBH_m"])
        if cbh_label > 100:  # obvious data-entry error (e.g. P01 tree 9 = 1202.00)
            print(f"{plot_name} tree {tid}: skipping, CBH_m={cbh_label} looks invalid", flush=True)
            continue

        mask = tree_mask_all & (pred_inst == tid)
        n = int(mask.sum())
        if n == 0:
            print(f"{plot_name} tree {tid}: no points found, skipping", flush=True)
            continue

        txyz = xyz_all[mask]
        tint = intensity[mask]
        hag = dtm.height_above_ground(txyz)
        tree_height = float(np.nanpercentile(hag, cfg.tree_height_percentile))

        all_tree_rows.append({
            "plot": plot_name, "tree_id": tid, "n_points": n,
            "height_m": tree_height, "CBH_m": cbh_label,
            "dbh_cm": float(row["dbh_cm"]) if "dbh_cm" in row else np.nan,
            "crown_base_height_m_old": float(row["crown_base_height_m"]) if "crown_base_height_m" in row else np.nan,
        })

        bin_edges = np.arange(0, tree_height + BIN_M, BIN_M)
        n_bins = bin_edges.shape[0] - 1
        bin_idx = np.clip(np.digitize(hag, bin_edges) - 1, 0, n_bins - 1)

        cell_x = np.floor(txyz[:, 0] / CELL_M).astype(np.int64)
        cell_y = np.floor(txyz[:, 1] / CELL_M).astype(np.int64)

        for i in range(n_bins):
            sel = bin_idx == i
            cnt = int(sel.sum())
            if cnt == 0:
                continue
            cx = cell_x[sel]
            cy = cell_y[sel]
            n_occupied_cells = len(np.unique(np.column_stack([cx, cy]), axis=0))
            area_m2 = n_occupied_cells * (CELL_M ** 2)
            inten_vals = tint[sel]

            all_bin_rows.append({
                "plot": plot_name, "tree_id": tid,
                "bin_low": float(bin_edges[i]), "bin_high": float(bin_edges[i + 1]),
                "hag_mid": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "n_points": cnt, "footprint_area_m2": area_m2,
                "intensity_median": float(np.median(inten_vals)),
                "intensity_mean": float(np.mean(inten_vals)),
            })

        print(f"{plot_name} tree {tid}: n={n} height={tree_height:.2f} CBH_m={cbh_label} -> {n_bins} bins", flush=True)

bin_df = pd.DataFrame(all_bin_rows)
tree_df = pd.DataFrame(all_tree_rows)
bin_df.to_csv(OUT_BINS, index=False)
tree_df.to_csv(OUT_TREES, index=False)
print(f"\nWrote {len(bin_df)} bin rows for {len(tree_df)} trees", flush=True)
print("ALL DONE", flush=True)
