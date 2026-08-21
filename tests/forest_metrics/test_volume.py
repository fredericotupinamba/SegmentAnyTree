import numpy as np
import pandas as pd

from tupisat_inference.forest_metrics.config import ForestMetricsConfig, LogAssortment
from tupisat_inference.forest_metrics.volume import (
    smalian_frustum_volume_m3,
    integrate_taper_to_volume_m3,
    conic_volume_estimate_m3,
    volume_by_log_assortment,
)


def _linear_taper_df(base_diameter_cm, height_m, increment=0.5):
    heights = np.arange(0.1, height_m, increment)
    diameters = base_diameter_cm * (1 - heights / height_m)
    return pd.DataFrame({
        "tree_id": 1,
        "height_m": heights,
        "diameter_cm": diameters,
        "cci": 1.0,
        "n_points": 20,
    })


def test_smalian_matches_cylinder_volume_for_equal_diameters():
    v = smalian_frustum_volume_m3(d_bottom_cm=20, d_top_cm=20, length_m=2.0)
    expected = np.pi * (0.10 ** 2) * 2.0
    assert abs(v - expected) < 1e-9


def test_integrate_taper_matches_closed_form_cone_volume():
    height_m = 20.0
    base_diameter_cm = 40.0
    taper_df = _linear_taper_df(base_diameter_cm, height_m)

    v = integrate_taper_to_volume_m3(taper_df, height_m, ForestMetricsConfig())
    expected = (np.pi / 3) * (base_diameter_cm / 200) ** 2 * height_m

    assert np.isfinite(v)
    assert abs(v - expected) / expected < 0.05


def test_integrate_taper_empty_curve_is_nan():
    cfg = ForestMetricsConfig()
    empty = pd.DataFrame(columns=["tree_id", "height_m", "diameter_cm", "cci", "n_points"])
    assert np.isnan(integrate_taper_to_volume_m3(empty, 20.0, cfg))


def test_conic_volume_estimate_matches_formula():
    v = conic_volume_estimate_m3(dbh_cm=30.0, height_m=15.0)
    expected = (np.pi / 3) * (0.15 ** 2) * 15.0
    assert abs(v - expected) < 1e-9


def test_conic_volume_estimate_nan_when_inputs_missing():
    assert np.isnan(conic_volume_estimate_m3(np.nan, 15.0))
    assert np.isnan(conic_volume_estimate_m3(30.0, np.nan))


def test_volume_by_log_assortment_cuts_expected_log_count():
    cfg = ForestMetricsConfig(
        log_assortments=[
            LogAssortment(name="sawlog", min_top_diameter_cm=20.0, log_length_m=4.0),
            LogAssortment(name="pulpwood", min_top_diameter_cm=8.0, log_length_m=3.0),
        ]
    )
    height_m = 20.0
    taper_df = _linear_taper_df(base_diameter_cm=40.0, height_m=height_m)

    result = volume_by_log_assortment(taper_df, height_m, cfg)

    # Linear taper from 40cm at base to 0 at 20m: diameter(h) = 40*(1-h/20).
    # Sawlogs (4m each) cut while top diameter >= 20cm, i.e. while
    # 40*(1-h/20) >= 20 -> h <= 10. So 2 sawlogs (0-4, 4-8) fit fully;
    # the 3rd (8-12) has top diameter at h=12 -> 40*(1-12/20)=16 < 20, stops.
    assert result["log_count_sawlog"] == 2
    assert result["volume_sawlog_m3"] > 0
    assert result["merchantable_volume_m3"] >= result["volume_sawlog_m3"]


def test_volume_by_log_assortment_zero_logs_for_thin_tree():
    cfg = ForestMetricsConfig(
        log_assortments=[LogAssortment(name="sawlog", min_top_diameter_cm=20.0, log_length_m=4.0)]
    )
    height_m = 5.0
    taper_df = _linear_taper_df(base_diameter_cm=10.0, height_m=height_m)

    result = volume_by_log_assortment(taper_df, height_m, cfg)

    assert result["log_count_sawlog"] == 0
    assert result["volume_sawlog_m3"] == 0.0
    assert result["merchantable_volume_m3"] == 0.0
