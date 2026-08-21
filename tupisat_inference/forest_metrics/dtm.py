from typing import List, Tuple

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from tupisat_inference.forest_metrics.config import ForestMetricsConfig


def extract_ground_points(non_tree_xyz: np.ndarray, cfg: ForestMetricsConfig) -> np.ndarray:
    """Run a Cloth Simulation Filter over non-tree (PredSemantic==0) points to
    separate ground from noise/understory/low branches. Returns an Nx3 array
    of ground points, or an empty (0, 3) array if none were found or the
    input was empty."""
    if non_tree_xyz.shape[0] == 0:
        return np.zeros((0, 3))

    import CSF

    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cfg.csf_cloth_resolution
    csf.params.rigidness = cfg.csf_rigidness
    csf.params.time_step = cfg.csf_time_step
    csf.params.class_threshold = cfg.csf_class_threshold
    csf.params.iterations = cfg.csf_iterations

    csf.setPointCloud(non_tree_xyz.astype(np.double))
    ground_idx = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    # exportCloth defaults to True and writes a cloth_nodes.txt debug dump
    # into the current working directory on every call -- not wanted here.
    csf.do_filtering(ground_idx, non_ground_idx, exportCloth=False)

    if len(ground_idx) == 0:
        return np.zeros((0, 3))

    return non_tree_xyz[np.array(ground_idx, dtype=np.int64)]


def _grid_dtm(ground_xyz: np.ndarray, all_xyz: np.ndarray, cfg: ForestMetricsConfig) -> Tuple[np.ndarray, List[str]]:
    """Grid the plot's XY extent and estimate ground Z per cell from the
    dtm_min_points_per_cell nearest ground points, falling back to all_xyz
    for cells whose nearest ground points are all farther than max_radius
    (or when there are no ground points at all).

    Batched cKDTree queries (one call covering every grid cell) rather than
    a per-cell Python loop -- for a dense TLS/MLS plot with tens of
    thousands of grid cells, a per-cell query_ball_point loop with a growing
    search radius was the dominant cost of the whole forest_metrics step."""
    warnings: List[str] = []

    xmin, ymin = np.floor(all_xyz[:, :2].min(axis=0))
    xmax, ymax = np.ceil(all_xyz[:, :2].max(axis=0))
    x_edges = np.arange(xmin, xmax + cfg.dtm_grid_resolution, cfg.dtm_grid_resolution)
    y_edges = np.arange(ymin, ymax + cfg.dtm_grid_resolution, cfg.dtm_grid_resolution)
    gx, gy = np.meshgrid(x_edges, y_edges)
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    n_cells = grid_xy.shape[0]

    max_radius = cfg.dtm_grid_resolution * 20
    z = np.full(n_cells, np.nan)
    needs_fallback = np.ones(n_cells, dtype=bool)

    if ground_xyz.shape[0] > 0:
        k = min(cfg.dtm_min_points_per_cell, ground_xyz.shape[0])
        ground_tree = cKDTree(ground_xyz[:, :2])
        dists, idx = ground_tree.query(grid_xy, k=k)
        if k == 1:
            dists, idx = dists[:, None], idx[:, None]
        well_served = dists[:, -1] <= max_radius
        z[well_served] = np.percentile(ground_xyz[idx[well_served], 2], cfg.dtm_ground_percentile, axis=1)
        needs_fallback = ~well_served

    fallback_cells = int(needs_fallback.sum())
    if fallback_cells > 0:
        all_tree = cKDTree(all_xyz[:, :2])
        k2 = min(cfg.dtm_min_points_per_cell, all_xyz.shape[0])
        dists2, idx2 = all_tree.query(grid_xy[needs_fallback], k=k2)
        if k2 == 1:
            dists2, idx2 = dists2[:, None], idx2[:, None]
        z[needs_fallback] = np.percentile(all_xyz[idx2, 2], cfg.dtm_fallback_percentile, axis=1)
        warnings.append(
            f"{fallback_cells}/{n_cells} DTM grid cells had no nearby ground points; "
            "fell back to a low percentile of all points in those cells."
        )

    return np.column_stack([grid_xy, z]), warnings


class DTM:
    def __init__(self, dtm_grid_xyz: np.ndarray):
        if dtm_grid_xyz.shape[0] == 0:
            raise ValueError("Cannot build a DTM from zero grid points.")
        self._xy = dtm_grid_xyz[:, :2]
        self._z = dtm_grid_xyz[:, 2]
        self._tree = cKDTree(self._xy)

    def ground_z(self, xy: np.ndarray, k: int = 3) -> np.ndarray:
        xy = np.atleast_2d(xy)
        k = min(k, self._xy.shape[0])
        dist, idx = self._tree.query(xy, k=k)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)

        # Points that land exactly on a grid vertex get zero distance; avoid
        # a division by zero in the inverse-distance weights.
        dist = np.where(dist == 0, 1e-9, dist)
        weights = 1.0 / dist
        weights /= weights.sum(axis=1, keepdims=True)
        z = np.sum(weights * self._z[idx], axis=1)
        return z

    def height_above_ground(self, xyz: np.ndarray) -> np.ndarray:
        return xyz[:, 2] - self.ground_z(xyz[:, :2])

    def convex_hull_area_m2(self) -> float:
        if self._xy.shape[0] < 3:
            return 0.0
        return ConvexHull(self._xy).volume

    @property
    def xy_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._xy.min(axis=0), self._xy.max(axis=0)


def build_dtm(non_tree_xyz: np.ndarray, all_xyz: np.ndarray, cfg: ForestMetricsConfig) -> Tuple[DTM, List[str]]:
    """Top-level DTM builder: CSF ground extraction -> gridding -> DTM.
    Falls back to a flat DTM from all_xyz if non_tree_xyz is empty. Never
    raises for data-quality problems; returns (dtm, warnings)."""
    warnings: List[str] = []

    if non_tree_xyz.shape[0] == 0:
        warnings.append(
            "No non-tree points available for ground extraction; falling back to a flat "
            "DTM derived from all points. Height-above-ground values for this file are suspect."
        )
        z = np.percentile(all_xyz[:, 2], cfg.dtm_fallback_percentile)
        (xmin, ymin), (xmax, ymax) = all_xyz[:, :2].min(axis=0), all_xyz[:, :2].max(axis=0)
        flat_grid = np.array([
            [xmin, ymin, z], [xmax, ymin, z], [xmin, ymax, z], [xmax, ymax, z],
        ])
        return DTM(flat_grid), warnings

    ground_xyz = extract_ground_points(non_tree_xyz, cfg)
    if ground_xyz.shape[0] == 0:
        warnings.append(
            "CSF found no ground points; falling back to a flat DTM derived from non-tree "
            "points. Height-above-ground values for this file are suspect."
        )
        z = np.percentile(non_tree_xyz[:, 2], cfg.dtm_fallback_percentile)
        (xmin, ymin), (xmax, ymax) = all_xyz[:, :2].min(axis=0), all_xyz[:, :2].max(axis=0)
        flat_grid = np.array([
            [xmin, ymin, z], [xmax, ymin, z], [xmin, ymax, z], [xmax, ymax, z],
        ])
        return DTM(flat_grid), warnings

    grid_xyz, grid_warnings = _grid_dtm(ground_xyz, all_xyz, cfg)
    warnings.extend(grid_warnings)
    return DTM(grid_xyz), warnings
