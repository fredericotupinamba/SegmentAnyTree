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
    # Circle fit + validation, ported from 3DFin/dendromatics'
    # sections.py (fit_circle / inner_circle / sector_occupancy /
    # fit_circle_check). Defaults below are 3DFin's own published defaults
    # (AdvancedParameters/ExpertParameters in three_d_fin.processing.
    # configuration), except min_points_for_circle_fit above -- 3DFin
    # defaults that gate to 80 points minimum, a hard rejection rather than
    # a soft one, and our TLS slice density hasn't been compared against
    # theirs yet.
    # Proportion of the fitted radius used for the inner-circle check: a fit
    # is suspect if too many points end up *inside* a circle this much
    # smaller than the fitted one (points bunched near the center rather
    # than on the ring).
    circle_diameter_proportion: float = 0.5
    # Max points allowed inside the inner circle before the fit is flagged.
    circle_inner_points_threshold: int = 5
    # Max distance between points to be considered part of the same cluster
    # when falling back to spatial clustering after a failed fit (isolates
    # the stem ring from e.g. a branch/foliage cluster in the same slice).
    circle_max_point_distance_m: float = 0.02
    # Angular sectors the fitted ring is divided into, and how many must
    # contain a point (within circle_sector_width_m of the fitted radius)
    # for the fit to be considered well-covered.
    circle_n_sectors: int = 16
    circle_min_occupied_sectors: int = 9
    circle_sector_width_m: float = 0.02
    # Sanity floor on fitted radius -- distinct from min_valid_dbh_cm below,
    # which rejects a whole *tree* later; this just rejects a degenerate
    # near-zero fit at the single-slice level.
    circle_min_radius_m: float = 0.02
    min_cci_for_valid_dbh: float = 0.3
    # A circle fit to a sparse or near-collinear slice (e.g. a branch stub,
    # or a stem partly occluded from the scanner) can converge on an
    # enormous, physically impossible radius. No real tree stem exceeds
    # this; reject fits above it rather than reporting nonsense.
    max_plausible_diameter_cm: float = 250.0
    # tilt_detection (ported from dendromatics/sections.py): flags a
    # section whose fitted circle *center* deviates too much from the
    # tree's other section centers (e.g. the fit was pulled sideways by a
    # branch cluster) -- a different failure mode than radius/coverage
    # quality, which inner_circle/sector_occupancy alone can miss. Score is
    # a weighted sum of absolute + relative tilt outlier flags, 0-1 scale;
    # sections above this are treated as unreliable, same as low-CCI ones.
    tilt_outlier_threshold: float = 0.5
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

    # Crown metrics. crown_base_height_m is found by a bottom-up scan of
    # height bins, looking for the first sustained run where LiDAR
    # intensity has dropped relative to the tree's own trunk baseline AND
    # the horizontal footprint area has grown relative to it. Calibrated
    # against 58 real, visually-labeled crown base heights across two
    # plots (P01/P02): mean absolute error 1.26m fit on all 58 trees,
    # 1.29m/1.53m under leave-one-plot-out cross-validation, vs. 3.37m for
    # the point-density-only heuristic this replaced (a densely-scanned
    # TLS tree can have roughly uniform point density top to bottom, so
    # density alone finds false transitions partway up the trunk).
    # Intensity separates real trunk (bark) from foliage far better than
    # local point geometry -- a from-scratch sphericity-based classifier
    # was tried first and found to not discriminate wood from foliage at
    # all on this real data (near-identical distributions in both
    # regions); intensity showed a real, consistent gap (trunk-base
    # median 2.7-4.4x higher than crown-top median across sample trees).
    # A third condition (point density above its own trunk baseline) was
    # tested too -- it also increases into the crown, but adding it as a
    # requirement was not consistently better across the two calibration
    # plots (helped one direction of cross-validation, hurt the other),
    # so it's left out in favor of the simpler two-feature rule.
    crown_height_bin_m: float = 0.25
    # Horizontal grid cell size for the footprint-area proxy: number of
    # occupied cells at this resolution x cell area, not a convex hull --
    # robust to a single stray branch point inflating the footprint the
    # way a hull's outer boundary would.
    crown_footprint_cell_m: float = 0.10
    # Minimum points in a bin to trust its intensity/footprint-area value
    # when *evaluating* whether that bin is crown -- an emptier bin is
    # skipped without breaking the current consecutive-bin run, not
    # treated as evidence either way.
    crown_min_points_per_bin: int = 10
    # Minimum points in a trunk-region bin (height <= dbh_height_m) to
    # trust it when computing the tree's own trunk intensity/area
    # baseline; falls back to using all trunk-region bins if none clear
    # this (only for very sparse trees).
    crown_baseline_min_points_per_bin: int = 15
    # A bin counts as crown once its median intensity has dropped below
    # this fraction of the tree's trunk-baseline intensity...
    crown_intensity_ratio_threshold: float = 0.4
    # ...AND its footprint area has grown beyond this multiple of the
    # tree's trunk-baseline footprint area.
    crown_area_ratio_threshold: float = 2.5
    crown_min_consecutive_bins: int = 5
    crown_voxel_size_m: float = 0.25

    # Taper / volume. No fixed max height: sampling always stops at each
    # tree's own measured height. 0.1 + n*0.2 lands exactly on 1.3m (DBH
    # height), so the taper grid always includes a sample at breast height
    # alongside the dedicated compute_dbh measurement there.
    taper_height_min_m: float = 0.1
    taper_height_increment_m: float = 0.2
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
