import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _spatial_cluster_labels(xy: np.ndarray, max_dist: float) -> np.ndarray:
    """Connected-components clustering at a fixed distance threshold --
    equivalent to single-linkage clustering (e.g. scipy's fclusterdata),
    but scalable: a KD-tree finds only the point pairs actually within
    max_dist instead of materializing a full O(n^2) pairwise distance
    matrix, which measured at hundreds of GiB for a real dense base slice
    with hundreds of thousands of points (see caller for the voxelization
    that bounds the input further still)."""
    n = xy.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)
    pairs = cKDTree(xy).query_pairs(max_dist, output_type="ndarray")
    if pairs.shape[0] == 0:
        return np.arange(n)
    graph = coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def split_merged_instances(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    semantic_col: str = "PredSemantic",
    instance_col: str = "PredInstance",
    base_slice_thickness_m: float = 1.0,
    voxel_size_m: float = 0.05,
    stem_cluster_distance_m: float = 0.3,
    min_stem_cluster_points: int = 50,
    min_stem_separation_m: float = 1.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Splits a predicted tree instance that actually contains two or more
    physically separate stems into separate instance IDs.

    The instance segmentation network clusters points per inference block
    and merges across blocks by point-ID IoU -- nothing in that pipeline
    checks whether a resulting instance is spatially contiguous, so two
    adjacent trees whose canopies overlap can end up sharing one instance
    ID (verified on real data: two stems 3.6m apart at the base, fused into
    one instance, tanked its measured diameters downstream).

    For each instance, this looks at a horizontal slice near its own lowest
    point (a per-instance proxy for "near the ground" -- no DTM/normalized
    height exists yet at this stage of the pipeline) and checks whether
    those base points form two or more spatially separate, large-enough
    clusters. If so, every point of the instance (not just the base slice)
    is reassigned to its nearest base-cluster centroid in XY -- the same
    nearest-vertical-axis idea dendromatics' individualize_trees uses.

    Known limitation, accepted rather than solved here: a genuinely
    multi-stemmed single tree (coppice growth) looks the same as two
    merged trees from base-point spacing alone, and would also get split.
    Thresholds default conservative (favor missing a real merge over
    splitting a real single tree) since a missed merge is already partly
    mitigated downstream by CCI/tilt-based quality gates in forest_metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain x/y/z coordinate columns and semantic/instance columns
        (names configurable via *_col arguments, to support both the
        internal merge dataframe's lowercase columns and an already-merged
        LAS/LAZ's uppercase columns).
    verbose : bool
        Print how many instances were split.

    Returns
    -------
    pd.DataFrame
        `df` with `instance_col` updated in place for split instances
        (also returned for convenience). Points outside tree_mask
        (non-tree semantic, or instance_col <= 0) are never touched.
    """
    tree_mask = (df[semantic_col] == 1) & (df[instance_col] > 0)
    if not tree_mask.any():
        return df

    next_id = int(df.loc[tree_mask, instance_col].max()) + 1
    n_split = 0

    for instance_id, idx in df.loc[tree_mask].groupby(instance_col).groups.items():
        pts = df.loc[idx, [x_col, y_col, z_col]]
        z_min = pts[z_col].min()
        base_mask = pts[z_col] <= z_min + base_slice_thickness_m
        base_xy = pts.loc[base_mask, [x_col, y_col]].to_numpy()
        if base_xy.shape[0] < min_stem_cluster_points * 2:
            continue

        # fclusterdata's pairwise distance matrix is O(n^2) in memory -- a
        # dense TLS base slice can have hundreds of thousands of points,
        # which blows up (verified: one real instance's base slice needed
        # 317 GiB as raw points). Clustering unique voxel centers instead
        # bounds the input to the slice's spatial footprint (at most a few
        # thousand cells), independent of how many raw points it has.
        voxel_idx = np.floor(base_xy / voxel_size_m).astype(np.int64)
        voxel_xy, inverse, voxel_counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)
        if voxel_xy.shape[0] < 2:
            continue
        voxel_centers = (voxel_xy + 0.5) * voxel_size_m

        cluster_id = _spatial_cluster_labels(voxel_centers, stem_cluster_distance_m)
        labels = np.unique(cluster_id)

        # Point-weighted count per cluster (a voxel's weight is how many
        # real points fell in it), not just number of occupied voxels.
        label_point_counts = {lbl: int(voxel_counts[cluster_id == lbl].sum()) for lbl in labels}
        big_enough = [lbl for lbl in labels if label_point_counts[lbl] >= min_stem_cluster_points]
        if len(big_enough) < 2:
            continue

        centroids = np.array(
            [np.average(voxel_centers[cluster_id == lbl], axis=0, weights=voxel_counts[cluster_id == lbl]) for lbl in big_enough]
        )
        if pdist(centroids).min() < min_stem_separation_m:
            continue

        all_xy = pts[[x_col, y_col]].to_numpy()
        dists = np.linalg.norm(all_xy[:, None, :] - centroids[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)

        n_new_stems = len(big_enough)
        new_ids = np.array([instance_id] + list(range(next_id, next_id + n_new_stems - 1)))
        next_id += n_new_stems - 1
        df.loc[idx, instance_col] = new_ids[nearest]
        n_split += 1

    if verbose and n_split:
        print(f"[split_merged_instances] Split {n_split} merged instance(s) into separate trees.")

    return df
