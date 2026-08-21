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
)

TAPER_COLUMNS = ["tree_id", "height_m", "diameter_cm", "cci", "n_points", "center_x", "center_y"]

REQUIRED_COLUMNS = ("X", "Y", "Z", "PredSemantic", "PredInstance")


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
        df = df[list(REQUIRED_COLUMNS)].copy()

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
        n_trees = tree_pts_df["PredInstance"].nunique() if not tree_pts_df.empty else 0
        self._log(f"Measuring {n_trees} tree(s)")

        for tree_id, group in tree_pts_df.groupby("PredInstance"):
            row, taper_df = measure_tree(int(tree_id), group, dtm, self.cfg)
            row.update(volume_by_log_assortment(taper_df, row["height_m"], self.cfg))
            row["stem_volume_taper_m3"] = integrate_taper_to_volume_m3(taper_df, row["height_m"], self.cfg)
            row["stem_volume_conic_m3"] = conic_volume_estimate_m3(row["dbh_cm"], row["height_m"])
            rows.append(row)
            taper_frames.append(taper_df)

        tree_metrics_df = pd.DataFrame(rows)
        taper_df_all = (
            pd.concat(taper_frames, ignore_index=True) if taper_frames else pd.DataFrame(columns=TAPER_COLUMNS)
        )

        plot_summary = summarize_plot(tree_metrics_df, dtm, tree_pts_df, self.cfg)
        if dtm_warnings:
            plot_summary["dtm_quality"] = "suspect"

        self._log("Building visualization point clouds")
        base_markers_df = build_base_marker_points(tree_metrics_df, self.cfg)
        diameter_circles_df = build_diameter_circle_points(tree_metrics_df, taper_df_all, dtm, self.cfg)
        labels_df = build_label_points(tree_metrics_df, dtm, self.cfg)

        self._write_outputs(tree_metrics_df, taper_df_all, plot_summary, base_markers_df, diameter_circles_df, labels_df)
        self._log(f"Wrote outputs for {n_trees} tree(s) to {self.output_dir}")

    def _write_outputs(self, tree_metrics_df, taper_df_all, plot_summary, base_markers_df, diameter_circles_df, labels_df):
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
