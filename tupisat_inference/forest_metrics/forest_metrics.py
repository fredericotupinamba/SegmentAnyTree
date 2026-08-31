import argparse
import json
import os
import sys

# Allows running this file directly (`python tupisat_inference/forest_metrics/
# forest_metrics.py ...`) without PYTHONPATH pre-set -- mirrors the same
# bootstrap in batch_orchestrator.py, needed because run_subprocess()
# normally relies on the orchestrator's own inherited PYTHONPATH.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

from tupisat_inference.las_to_pandas import las_to_pandas
from tupisat_inference.pandas_to_las import pandas_to_las
from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import build_dtm
from tupisat_inference.forest_metrics.tree_metrics import measure_tree
from tupisat_inference.forest_metrics.volume import integrate_taper_to_volume_m3, conic_volume_estimate_m3, volume_by_log_assortment
from tupisat_inference.forest_metrics.stand_metrics import summarize_plot
from tupisat_inference.forest_metrics.visualization import (
    build_base_marker_points,
    build_diameter_circle_points,
    build_label_points,
    write_crown_classification_laz,
)

TAPER_COLUMNS = [
    "tree_id", "height_m", "diameter_cm", "cci", "n_points", "center_x", "center_y",
    "tilt_outlier_prob", "axis_residual_m", "fit_source", "diameter_corrected_cm",
]

REQUIRED_COLUMNS = ("X", "Y", "Z", "intensity", "PredSemantic", "PredInstance")

# Written by PointsToWood (0.0 = leaf, 1.0 = wood). Optional: without it
# compute_crown_metrics falls back to its intensity+area rule, which is
# less accurate (1.09m vs 0.71m cross-validated) but needs no extra input.
OPTIONAL_COLUMNS = ("prediction",)


class ForestMetrics(object):
    def __init__(self, input_las_path, output_dir, stem=None, config_json=None, verbose=False):
        self.input_las_path = input_las_path
        self.output_dir = output_dir
        self.stem = stem or os.path.splitext(os.path.basename(input_las_path))[0]
        self.cfg = ForestMetricsConfig.from_json(config_json)
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[forest_metrics] {msg}")

    def run(self):
        if not os.path.isfile(self.input_las_path) or os.path.getsize(self.input_las_path) == 0:
            raise ValueError(f"Input point cloud missing or empty: {self.input_las_path}")

        self._log(f"Loading {self.input_las_path}")
        df = las_to_pandas(self.input_las_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.input_las_path} is missing required column(s) {missing}; "
                "expected a SAT merged output (with PredSemantic/PredInstance)."
            )
        # Merged files carry ~20 LAS dimensions we never use (intensity,
        # gps_time, rgb, ...); a 90M-point file loaded at full width can peak
        # at tens of GB. Drop everything but what this module needs before
        # doing any further work.
        present_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
        df = df[list(REQUIRED_COLUMNS) + present_optional].copy()
        if "prediction" in present_optional:
            self._log("Found PointsToWood wood/leaf labels -- using the wood-fraction crown rule")
        else:
            self._log("No 'prediction' column -- falling back to the intensity+area crown rule")

        non_tree_df = df[df["PredSemantic"] == 0]
        tree_pts_df = df[(df["PredSemantic"] == 1) & (df["PredInstance"] > 0)]

        self._log(f"{len(non_tree_df)} non-tree points, {len(tree_pts_df)} tree points")
        dtm, dtm_warnings = build_dtm(
            non_tree_df[["X", "Y", "Z"]].values, df[["X", "Y", "Z"]].values, self.cfg
        )
        for w in dtm_warnings:
            print(f"WARNING: {w}")

        rows = []
        taper_frames = []
        # Aligned 1:1 with df's (and so the original file's) point order --
        # scattered into per-tree during the loop below, left at 0 (not
        # crown) for every non-tree point and every tree point below its
        # own crown_base_height_m.
        is_crown_full = np.zeros(len(df), dtype=np.uint8)
        n_instances = tree_pts_df["PredInstance"].nunique() if not tree_pts_df.empty else 0
        self._log(f"Measuring {n_instances} segmented instance(s)")

        for tree_id, group in tree_pts_df.groupby("PredInstance"):
            row, taper_df, is_crown_point = measure_tree(int(tree_id), group, dtm, self.cfg)
            is_crown_full[group.index.to_numpy()] = is_crown_point.astype(np.uint8)
            row.update(volume_by_log_assortment(taper_df, row["height_m"], self.cfg, diameter_column="diameter_corrected_cm"))
            row["stem_volume_taper_m3"] = integrate_taper_to_volume_m3(
                taper_df, row["height_m"], self.cfg, diameter_column="diameter_corrected_cm"
            )
            row["stem_volume_conic_m3"] = conic_volume_estimate_m3(row["dbh_cm"], row["height_m"])
            rows.append(row)
            taper_frames.append(taper_df)

        tree_metrics_df = pd.DataFrame(rows)
        taper_df_all = (
            pd.concat(taper_frames, ignore_index=True) if taper_frames else pd.DataFrame(columns=TAPER_COLUMNS)
        )

        # Segmented instances that failed the tree-vs-shrub/plot-edge-fragment
        # check (measure_tree's is_valid_tree) don't represent a countable
        # tree in the plot -- drop them entirely rather than including them,
        # flagged, in the outputs.
        n_trees = 0
        if not tree_metrics_df.empty:
            valid_mask = tree_metrics_df["is_valid_tree"]
            n_rejected = int((~valid_mask).sum())
            if n_rejected:
                self._log(f"Rejected {n_rejected} segmented instance(s) as not a tree (shrub/plot-edge fragment)")
            valid_ids = tree_metrics_df.loc[valid_mask, "tree_id"]
            tree_metrics_df = tree_metrics_df[valid_mask].drop(columns=["is_valid_tree"]).reset_index(drop=True)
            taper_df_all = taper_df_all[taper_df_all["tree_id"].isin(valid_ids)].reset_index(drop=True)
            n_trees = len(tree_metrics_df)

        plot_summary = summarize_plot(tree_metrics_df, dtm, tree_pts_df, self.cfg)
        if dtm_warnings:
            plot_summary["dtm_quality"] = "suspect"

        self._log("Building visualization point clouds")
        base_markers_df = build_base_marker_points(tree_metrics_df, self.cfg)
        diameter_circles_df = build_diameter_circle_points(tree_metrics_df, taper_df_all, dtm, self.cfg)
        labels_df = build_label_points(tree_metrics_df, dtm, self.cfg)

        self._write_outputs(
            tree_metrics_df, taper_df_all, plot_summary, base_markers_df, diameter_circles_df, labels_df, is_crown_full
        )
        self._log(f"Wrote outputs for {n_trees} tree(s) to {self.output_dir}")

    def _write_outputs(
        self, tree_metrics_df, taper_df_all, plot_summary, base_markers_df, diameter_circles_df, labels_df, is_crown_full
    ):
        os.makedirs(self.output_dir, exist_ok=True)
        tree_metrics_df.to_csv(os.path.join(self.output_dir, f"{self.stem}_tree_metrics.csv"), index=False)
        taper_df_all.to_csv(os.path.join(self.output_dir, f"{self.stem}_taper.csv"), index=False)
        pd.DataFrame([plot_summary]).to_csv(os.path.join(self.output_dir, f"{self.stem}_plot_summary.csv"), index=False)
        with open(os.path.join(self.output_dir, f"{self.stem}_plot_summary.json"), "w") as f:
            json.dump(plot_summary, f, indent=2)

        # Point-cloud visualizations, FSCT-style. Skip writing a file for an
        # empty cloud (e.g. no trees at all) rather than handing laspy a
        # zero-row DataFrame it can't compute bounds/offsets from.
        for name, df in (
            ("tree_bases", base_markers_df),
            ("diameter_circles", diameter_circles_df),
            ("tree_labels", labels_df),
        ):
            if df.empty:
                continue
            pandas_to_las(
                df,
                output_file_path=os.path.join(self.output_dir, f"{self.stem}_{name}.las"),
                do_compress=True,
            )

        # Adds an IsCrown (0/1) scalar field directly onto the input point
        # cloud (overwriting it in place) -- lets the crown/not-crown split
        # from compute_crown_metrics be inspected directly in a point cloud
        # viewer, the same way PredSemantic/PredInstance already can be.
        # Writes to the same path it read from rather than a separate
        # "_crown_classified.laz" copy: the two were near-duplicates of a
        # very large file (the full point cloud), doubling disk usage for
        # no benefit once the classification is trustworthy enough to want
        # by default.
        write_crown_classification_laz(self.input_las_path, is_crown_full)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-tree and plot-level forest inventory metrics from a "
        "SAT merged LAS/LAZ file (must contain PredSemantic/PredInstance columns)."
    )
    parser.add_argument("--input-las", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--config-json", default=None, help="Optional JSON file overriding ForestMetricsConfig defaults.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    ForestMetrics(args.input_las, args.output_dir, args.stem, args.config_json, args.verbose).run()
