import json
from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional


@dataclass
class LogAssortment:
    name: str
    min_top_diameter_cm: float
    log_length_m: float


def _default_log_assortments() -> List[LogAssortment]:
    return [
        LogAssortment("sawlog", 20.0, 4.0),
        LogAssortment("pulpwood", 8.0, 3.0),
    ]


@dataclass
class ForestMetricsConfig:
    # Ground / DTM (CSF defaults tuned for dense TLS/MLS point clouds, not airborne LiDAR).
    csf_cloth_resolution: float = 0.2
    csf_rigidness: int = 1
    csf_time_step: float = 0.65
    csf_class_threshold: float = 0.15
    csf_iterations: int = 500
    dtm_grid_resolution: float = 0.25
    dtm_min_points_per_cell: int = 20
    dtm_ground_percentile: float = 20.0
    dtm_fallback_percentile: float = 2.5

    # Per-tree geometry.
    min_points_per_tree: int = 30
    base_slice_thickness_m: float = 0.3
    dbh_height_m: float = 1.3
    dbh_slice_thickness_m: float = 0.1
    min_points_for_circle_fit: int = 10
    circle_ransac_residual_threshold_m: float = 0.02
    circle_ransac_min_samples_cap: int = 30
    dbh_ransac_max_trials: int = 1000
    taper_ransac_max_trials: int = 150
    min_cci_for_valid_dbh: float = 0.3
    # A least-squares circle fit to a sparse or near-collinear slice (e.g. a
    # branch stub, or a stem partly occluded from the scanner) can converge
    # on an enormous, physically impossible radius. No real tree stem
    # exceeds this; reject fits above it rather than reporting nonsense.
    max_plausible_diameter_cm: float = 250.0
    tree_height_percentile: float = 99.5

    # Tree-vs-not-a-tree classification. A segmented instance is only kept
    # as a tree if it reaches tree_validation_height_m at all (rejects
    # shrubs/undergrowth) AND shows a consistent, well-covered circular
    # cross-section at several heights between the ground and
    # tree_validation_height_m, not just one (rejects plot-edge crown
    # fragments and branch clusters that can look tree-like in a single
    # lucky slice but don't repeat). Verified against real TLS data that a
    # single slice's CCI is *not* reliable evidence on its own: three real,
    # well-formed trees (DBH CCI 0.85-1.0) got misclassified as not-a-tree
    # purely because their one dedicated slice at exactly
    # tree_validation_height_m had CCI 0.33-0.74 -- angular coverage
    # naturally degrades higher up a TLS/MLS scan, so one slice can dip
    # below threshold on a real tree for reasons that have nothing to do
    # with whether it's a shrub. tree_validation_cci_threshold (0.8) is
    # deliberately much stricter than min_cci_for_valid_dbh (0.3) -- that
    # one only asks "is this fit numerically sane", this one asks "is this
    # slice a real, fully-covered stem cross-section". DBH counts as one of
    # the qualifying slices if it clears the threshold.
    tree_validation_height_m: float = 4.0
    tree_validation_cci_threshold: float = 0.8
    min_high_cci_slices: int = 3
    min_valid_dbh_cm: float = 7.0

    # Crown metrics.
    crown_height_bin_m: float = 0.5
    crown_density_fraction: float = 0.3
    crown_min_consecutive_bins: int = 3
    crown_voxel_size_m: float = 0.25

    # Taper / volume. No fixed max height: sampling always stops at each
    # tree's own measured height.
    taper_height_min_m: float = 0.1
    taper_height_increment_m: float = 0.5
    taper_slice_thickness_m: float = 0.2
    min_points_for_taper_slice: int = 8
    log_assortments: List[LogAssortment] = field(default_factory=_default_log_assortments)

    # Stand-level.
    plot_area_mode: str = "convex_hull"  # or "fixed_radius"
    fixed_plot_radius_m: Optional[float] = None
    dbh_histogram_bin_cm: float = 5.0
    canopy_grid_resolution_m: float = 0.5
    canopy_min_points_per_cell: int = 5
    canopy_min_height_above_ground_m: float = 2.0

    # Point-cloud visualization outputs (diameter circles, text labels,
    # base markers) -- viewable in any point cloud viewer, FSCT-style.
    diameter_circle_n_points: int = 24
    # Each character is an 11x11-pixel bitmap, so this is real-world meters
    # PER PIXEL, not per character -- a 12-character line is ~132 pixels
    # wide. 0.003 gives roughly a 0.3-0.4 m wide line, comparable to FSCT's
    # own default (0.00256). Do not "round up" this value without checking
    # the resulting label width -- a value that looks small (e.g. 0.05) is
    # actually ~20x too large and produces multi-meter-wide garbled labels.
    label_character_size_m: float = 0.003
    label_line_height_m: float = 0.05
    base_marker_size_m: float = 0.15

    @staticmethod
    def from_json(path: Optional[str]) -> "ForestMetricsConfig":
        cfg = ForestMetricsConfig()
        if not path:
            return cfg

        with open(path, "r") as f:
            overrides = json.load(f)

        known_fields = {f.name for f in fields(cfg)}
        for key, value in overrides.items():
            if key not in known_fields:
                raise ValueError(f"Unknown ForestMetricsConfig field in {path}: {key}")
            if key == "log_assortments":
                value = [LogAssortment(**a) for a in value]
            setattr(cfg, key, value)

        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
