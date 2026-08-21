import glob
import os

import pandas as pd
import pytest

from tupisat_inference.forest_metrics.forest_metrics import ForestMetrics
from tupisat_inference.las_to_pandas import las_to_pandas

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(REPO_ROOT, "data", "output")

pytest.importorskip("CSF", reason="CSF (cloth-simulation-filter) not installed")


def _first_fixture():
    candidates = sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.laz")))
    if not candidates:
        pytest.skip(f"no .laz fixtures found in {FIXTURE_DIR}")
    return candidates[0]


def test_forest_metrics_end_to_end_against_real_output(tmp_path):
    input_las = _first_fixture()

    fm = ForestMetrics(input_las, str(tmp_path), stem="itest", verbose=True)
    fm.run()

    tree_metrics_path = tmp_path / "itest_tree_metrics.csv"
    taper_path = tmp_path / "itest_taper.csv"
    plot_summary_csv = tmp_path / "itest_plot_summary.csv"
    plot_summary_json = tmp_path / "itest_plot_summary.json"

    for p in (tree_metrics_path, taper_path, plot_summary_csv, plot_summary_json):
        assert p.exists(), f"expected output missing: {p}"
        assert p.stat().st_size > 0, f"output is empty: {p}"

    tree_metrics_df = pd.read_csv(tree_metrics_path)
    plot_summary_df = pd.read_csv(plot_summary_csv)

    assert "tree_id" in tree_metrics_df.columns
    assert "n_trees" in plot_summary_df.columns
    assert int(plot_summary_df["n_trees"].iloc[0]) == len(tree_metrics_df)

    # Visualization point clouds -- pandas_to_las(do_compress=True) swaps
    # the extension to .laz regardless of the .las path it's given.
    for name in ("tree_bases", "diameter_circles", "tree_labels"):
        path = tmp_path / f"itest_{name}.laz"
        assert path.exists(), f"expected visualization output missing: {path}"
        assert path.stat().st_size > 0, f"visualization output is empty: {path}"
        cloud = las_to_pandas(str(path))
        assert len(cloud) > 0
        assert "tree_id" in cloud.columns
