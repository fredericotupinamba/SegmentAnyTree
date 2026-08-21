import numpy as np
import pandas as pd

from tupisat_inference.forest_metrics.config import ForestMetricsConfig


def smalian_frustum_volume_m3(d_bottom_cm: float, d_top_cm: float, length_m: float) -> float:
    """Smalian's formula: V = (A_bottom + A_top) / 2 * length."""
    r_bottom_m = d_bottom_cm / 200.0
    r_top_m = d_top_cm / 200.0
    a_bottom = np.pi * r_bottom_m ** 2
    a_top = np.pi * r_top_m ** 2
    return (a_bottom + a_top) / 2 * length_m


def _fill_and_extrapolate_taper(taper_df: pd.DataFrame, tree_height: float, diameter_column: str = "diameter_cm") -> pd.DataFrame:
    """Linearly interpolate NaN diameters between valid heights; extrapolate
    the tip to diameter=0 at tree_height if missing; back-fill the base from
    the lowest valid diameter if missing.

    diameter_column defaults to the raw RANSAC diameter_cm (used directly by
    the tests below with hand-built curves); forest_metrics.py passes
    "diameter_corrected_cm" so volume is integrated over the
    monotonicity-corrected curve instead of a curve a stray branch slice
    could otherwise inflate mid-trunk."""
    df = taper_df[["height_m", diameter_column]].rename(columns={diameter_column: "diameter_cm"}).copy()
    if df["diameter_cm"].notna().sum() == 0:
        return df

    if not np.isclose(df["height_m"].iloc[-1], tree_height):
        df = pd.concat(
            [df, pd.DataFrame({"height_m": [tree_height], "diameter_cm": [np.nan]})],
            ignore_index=True,
        )

    if pd.isna(df["diameter_cm"].iloc[-1]):
        df.loc[df.index[-1], "diameter_cm"] = 0.0

    df["diameter_cm"] = df["diameter_cm"].interpolate(method="linear", limit_direction="both")
    return df


def integrate_taper_to_volume_m3(
    taper_df_single_tree: pd.DataFrame, tree_height: float, cfg: ForestMetricsConfig, diameter_column: str = "diameter_cm"
) -> float:
    if taper_df_single_tree.empty or not np.isfinite(tree_height):
        return np.nan

    filled = _fill_and_extrapolate_taper(taper_df_single_tree, tree_height, diameter_column)
    if filled["diameter_cm"].isna().all():
        return np.nan

    volume = 0.0
    heights = filled["height_m"].values
    diameters = filled["diameter_cm"].values
    for i in range(len(heights) - 1):
        length = heights[i + 1] - heights[i]
        if length <= 0:
            continue
        volume += smalian_frustum_volume_m3(diameters[i], diameters[i + 1], length)

    return volume


def conic_volume_estimate_m3(dbh_cm: float, height_m: float) -> float:
    if not np.isfinite(dbh_cm) or not np.isfinite(height_m) or dbh_cm <= 0 or height_m <= 0:
        return np.nan
    r_m = dbh_cm / 200.0
    return (np.pi / 3) * r_m ** 2 * height_m


def volume_by_log_assortment(
    taper_df_single_tree: pd.DataFrame, tree_height: float, cfg: ForestMetricsConfig, diameter_column: str = "diameter_cm"
) -> dict:
    result = {}
    for a in cfg.log_assortments:
        result[f"volume_{a.name}_m3"] = 0.0
        result[f"log_count_{a.name}"] = 0
    result["merchantable_volume_m3"] = 0.0

    if taper_df_single_tree.empty or not np.isfinite(tree_height):
        return result

    filled = _fill_and_extrapolate_taper(taper_df_single_tree, tree_height, diameter_column)
    if filled["diameter_cm"].isna().all():
        return result

    heights = filled["height_m"].values
    diameters = filled["diameter_cm"].values

    def diameter_at(h: float) -> float:
        if h <= heights[0]:
            return diameters[0]
        if h >= heights[-1]:
            return diameters[-1]
        return float(np.interp(h, heights, diameters))

    for a in cfg.log_assortments:
        current_h = 0.0
        while current_h + a.log_length_m <= tree_height:
            top_h = current_h + a.log_length_m
            d_bottom = diameter_at(current_h)
            d_top = diameter_at(top_h)
            if d_top >= a.min_top_diameter_cm:
                result[f"volume_{a.name}_m3"] += smalian_frustum_volume_m3(d_bottom, d_top, a.log_length_m)
                result[f"log_count_{a.name}"] += 1
                current_h = top_h
            else:
                break

    result["merchantable_volume_m3"] = sum(
        result[f"volume_{a.name}_m3"] for a in cfg.log_assortments
    )
    return result
