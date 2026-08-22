import os

import laspy
import numpy as np
import pandas as pd

from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import DTM

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "numbers")

# Character -> bitmap filename (ported from FSCT's tools/numbers/*.csv, an
# 11x11 0/1 grid per glyph). "M" and "m" are stored as separate files
# (_M.csv / m.csv) since filesystems that ignore case can't hold both
# "M.csv" and "m.csv" side by side.
_CHARACTER_FILES = {
    **{d: d for d in "0123456789"},
    **{c: c for c in "ABCDEFGHIJKLNOPQRSTUVWXYZ"},
    "M": "_M",
    "m": "m",
    ".": "dot",
    " ": "space",
    "_": "_",
    "-": "-",
    ":": "semiC",
}

_character_cache = {}


def _load_character(char: str) -> np.ndarray:
    filename = _CHARACTER_FILES.get(char, "space")
    if filename not in _character_cache:
        path = os.path.join(ASSETS_DIR, f"{filename}.csv")
        _character_cache[filename] = np.genfromtxt(path, delimiter=",")
    return _character_cache[filename]


def render_text_points(text: str, character_size: float, x: float, y: float, z: float) -> np.ndarray:
    """Renders `text` as a small 3D point-cloud label, tilted 45 degrees so
    it reads from an oblique viewing angle in a point cloud viewer (ported
    from FSCT's point_cloud_annotations)."""
    if not text:
        return np.zeros((0, 3))

    grids = [_load_character(c) for c in text]
    combined = np.hstack(grids)
    rotated = np.rot90(combined, axes=(1, 0))
    rows, cols = np.nonzero(rotated == 1)
    if rows.size == 0:
        return np.zeros((0, 3))
    points = np.column_stack([rows, cols, np.zeros_like(rows)]).astype(float)

    tilt = np.array([
        [1, 0, 0],
        [0, np.cos(-np.pi / 4), -np.sin(-np.pi / 4)],
        [0, np.sin(-np.pi / 4), np.cos(-np.pi / 4)],
    ])
    points = points @ tilt

    return points * character_size + [x, y, z]


def circle_points(xc: float, yc: float, z: float, radius_m: float, n_points: int) -> np.ndarray:
    """Ring of points representing a horizontal circle -- used to visualize
    a fitted stem diameter (DBH or a taper sample) in a point cloud viewer."""
    if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(z) and np.isfinite(radius_m)) or radius_m <= 0:
        return np.zeros((0, 3))
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = xc + radius_m * np.cos(angles)
    y = yc + radius_m * np.sin(angles)
    return np.column_stack([x, y, np.full(n_points, z)])


def cross_marker_points(x: float, y: float, z: float, size_m: float, points_per_arm: int = 7) -> np.ndarray:
    """Small 3D cross (one short line of points along each axis) marking a
    single location -- used to show where a tree's base was placed."""
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
        return np.zeros((0, 3))
    offsets = np.linspace(-size_m, size_m, points_per_arm)
    arm_x = np.column_stack([x + offsets, np.full(points_per_arm, y), np.full(points_per_arm, z)])
    arm_y = np.column_stack([np.full(points_per_arm, x), y + offsets, np.full(points_per_arm, z)])
    arm_z = np.column_stack([np.full(points_per_arm, x), np.full(points_per_arm, y), z + offsets])
    return np.vstack([arm_x, arm_y, arm_z])


def write_crown_classification_laz(las_path: str, is_crown: np.ndarray) -> None:
    """Adds an IsCrown scalar field (uint8, 0/1) directly onto `las_path`,
    overwriting it in place -- lets the crown/not-crown split be inspected
    directly in a point cloud viewer (CloudCompare, etc.), the same way
    PredSemantic/PredInstance already can be. `is_crown` must be aligned
    1:1 with the file's own point order (e.g. built from the same
    las_to_pandas DataFrame for that file, never reordered/filtered).
    Reads the whole file into memory before writing, so overwriting the
    same path it read from is safe (no read-while-writing race)."""
    las = laspy.read(las_path)
    if len(las.points) != is_crown.shape[0]:
        raise ValueError(
            f"is_crown length {is_crown.shape[0]} does not match point count "
            f"{len(las.points)} in {las_path}"
        )
    if "IsCrown" not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="IsCrown", type="uint8"))
    las.IsCrown = is_crown.astype(np.uint8)
    las.write(las_path)


def build_base_marker_points(tree_metrics_df: pd.DataFrame, cfg: ForestMetricsConfig) -> pd.DataFrame:
    rows = []
    for t in tree_metrics_df.itertuples():
        pts = cross_marker_points(t.x_base, t.y_base, t.z_base, cfg.base_marker_size_m)
        if pts.shape[0] == 0:
            continue
        df = pd.DataFrame(pts, columns=["x", "y", "z"])
        df["tree_id"] = np.int32(t.tree_id)
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["x", "y", "z", "tree_id"])
    return pd.concat(rows, ignore_index=True)


def build_diameter_circle_points(tree_metrics_df: pd.DataFrame, taper_df_all: pd.DataFrame, dtm: DTM, cfg: ForestMetricsConfig) -> pd.DataFrame:
    rows = []

    for t in tree_metrics_df.itertuples():
        if not (np.isfinite(t.dbh_x) and np.isfinite(t.dbh_y) and np.isfinite(t.dbh_cm)):
            continue
        z = float(dtm.ground_z(np.array([[t.dbh_x, t.dbh_y]]))[0]) + cfg.dbh_height_m
        pts = circle_points(t.dbh_x, t.dbh_y, z, t.dbh_cm / 200, cfg.diameter_circle_n_points)
        if pts.shape[0] == 0:
            continue
        df = pd.DataFrame(pts, columns=["x", "y", "z"])
        df["tree_id"] = np.int32(t.tree_id)
        df["diameter_cm"] = np.float32(t.dbh_cm)
        df["kind_code"] = np.uint8(0)  # 0 = DBH
        rows.append(df)

    if not taper_df_all.empty:
        # Use the monotonicity-corrected diameter for the visualized ring --
        # the raw RANSAC diameter_cm can still show a branch-inflated bulge
        # mid-trunk that apply_monotonic_correction (tree_metrics.py) has
        # already reconciled away.
        diameter_column = "diameter_corrected_cm" if "diameter_corrected_cm" in taper_df_all.columns else "diameter_cm"
        valid = taper_df_all.dropna(subset=[diameter_column, "center_x", "center_y"])
        for r in valid.itertuples():
            diameter_cm = getattr(r, diameter_column)
            z = float(dtm.ground_z(np.array([[r.center_x, r.center_y]]))[0]) + r.height_m
            pts = circle_points(r.center_x, r.center_y, z, diameter_cm / 200, cfg.diameter_circle_n_points)
            if pts.shape[0] == 0:
                continue
            df = pd.DataFrame(pts, columns=["x", "y", "z"])
            df["tree_id"] = np.int32(r.tree_id)
            df["diameter_cm"] = np.float32(diameter_cm)
            df["kind_code"] = np.uint8(1)  # 1 = taper sample
            rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["x", "y", "z", "tree_id", "diameter_cm", "kind_code"])
    return pd.concat(rows, ignore_index=True)


def build_label_points(tree_metrics_df: pd.DataFrame, dtm: DTM, cfg: ForestMetricsConfig) -> pd.DataFrame:
    """One small stacked-line label per tree (ID / DBH / height), placed
    beside the DBH ring at breast height when a valid DBH exists (matching
    where FSCT anchors its own text) -- falling back to just above the base
    for trees with no usable DBH, so every tree still gets an ID/height
    label somewhere legible."""
    rows = []
    for t in tree_metrics_df.itertuples():
        has_dbh = np.isfinite(t.dbh_x) and np.isfinite(t.dbh_y) and np.isfinite(t.dbh_cm)
        has_base = np.isfinite(t.x_base) and np.isfinite(t.y_base) and np.isfinite(t.z_base)
        if not has_dbh and not has_base:
            continue

        if has_dbh:
            anchor_x, anchor_y = t.dbh_x, t.dbh_y
            anchor_z = float(dtm.ground_z(np.array([[anchor_x, anchor_y]]))[0]) + cfg.dbh_height_m
            x_offset = t.dbh_cm / 200 + 0.15
        else:
            anchor_x, anchor_y, anchor_z = t.x_base, t.y_base, t.z_base
            x_offset = 0.3

        dbh_text = f"{t.dbh_cm:.1f}CM" if np.isfinite(t.dbh_cm) else "NA"
        height_text = f"{t.height_m:.1f}M" if np.isfinite(t.height_m) else "NA"
        lines = [
            f"ID:{int(t.tree_id)}",
            f"DBH:{dbh_text}",
            f"H:{height_text}",
        ]

        for i, line in enumerate(lines):
            z = anchor_z + (len(lines) - 1 - i) * cfg.label_line_height_m
            pts = render_text_points(line, cfg.label_character_size_m, anchor_x + x_offset, anchor_y, z)
            if pts.shape[0] == 0:
                continue
            df = pd.DataFrame(pts, columns=["x", "y", "z"])
            df["tree_id"] = np.int32(t.tree_id)
            rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["x", "y", "z", "tree_id"])
    return pd.concat(rows, ignore_index=True)
