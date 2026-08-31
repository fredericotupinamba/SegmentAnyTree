"""HISTORICAL -- not runnable as-is.

This is the calibration that produced the intensity+area crown rule and
its published 1.26m figure, kept so that comparison can be audited. Its
paths point at a scratch directory that no longer exists, and its feature
set predates the wood/leaf labels. The current, runnable versions are
extract_cbh_features.py and calibrate_cbh.py beside it.
"""
import itertools
import numpy as np
import pandas as pd

SC = "C:/Users/Fred/AppData/Local/Temp/claude/e--GITHUB-SegmentAnyTree/17ed6f96-d6d2-488a-946c-0426577ade70/scratchpad"
bins = pd.read_csv(f"{SC}/cbh_bin_features.csv")
trees = pd.read_csv(f"{SC}/cbh_tree_labels.csv")

MIN_POINTS_PER_BIN = 15

old_err = (trees["crown_base_height_m_old"] - trees["CBH_m"]).abs()
print(f"OLD density-only method: MAE={old_err.mean():.2f}m  median={old_err.median():.2f}m  "
      f"n_within_1m={int((old_err <= 1.0).sum())}/{len(trees)}", flush=True)

tree_base = {}
for (plot, tid), g in bins.groupby(["plot", "tree_id"]):
    base = g[g["hag_mid"] <= 1.3]
    base = base[base["n_points"] >= MIN_POINTS_PER_BIN]
    if base.empty:
        base = g[g["hag_mid"] <= 1.3]
    tree_base[(plot, tid)] = {
        "intensity0": base["intensity_median"].median(),
        "area0": base["footprint_area_m2"].median(),
        "density0": base["n_points"].median(),
    }

bins = bins.copy()
bins["intensity0"] = bins.apply(lambda r: tree_base[(r["plot"], r["tree_id"])]["intensity0"], axis=1)
bins["area0"] = bins.apply(lambda r: tree_base[(r["plot"], r["tree_id"])]["area0"], axis=1)
bins["density0"] = bins.apply(lambda r: tree_base[(r["plot"], r["tree_id"])]["density0"], axis=1)
bins["norm_intensity"] = bins["intensity_median"] / bins["intensity0"]
bins["norm_area"] = bins["footprint_area_m2"] / bins["area0"]
bins["norm_density"] = bins["n_points"] / bins["density0"]

# Precompute each tree's data as plain numpy arrays, sorted by height, once.
tree_arrays = {}
for (plot, tid), g in bins.groupby(["plot", "tree_id"]):
    g = g.sort_values("hag_mid")
    tree_arrays[(plot, tid)] = {
        "bin_low": g["bin_low"].to_numpy(),
        "n_points": g["n_points"].to_numpy(),
        "norm_intensity": g["norm_intensity"].to_numpy(),
        "norm_area": g["norm_area"].to_numpy(),
        "norm_density": g["norm_density"].to_numpy(),
    }

tree_keys = list(zip(trees["plot"], trees["tree_id"]))
truth_all = dict(zip(tree_keys, trees["CBH_m"]))


def predict_cbh_fast(arr, intensity_thr, area_thr, density_thr, min_run, min_points):
    n_pts = arr["n_points"]
    trust = n_pts >= min_points
    cond = (arr["norm_intensity"] < intensity_thr) & (arr["norm_area"] > area_thr)
    if density_thr is not None:
        cond = cond & (arr["norm_density"] > density_thr)

    run = 0
    run_start = None
    bin_low = arr["bin_low"]
    for i in range(len(bin_low)):
        if not trust[i]:
            continue
        if cond[i]:
            if run == 0:
                run_start = bin_low[i]
            run += 1
            if run >= min_run:
                return run_start
        else:
            run = 0
    return np.nan


def evaluate_fast(param, keys, use_density):
    intensity_thr, area_thr, density_thr, min_run, min_points = param
    if not use_density:
        density_thr = None
    errs = []
    n_valid = 0
    for k in keys:
        arr = tree_arrays[k]
        pred = predict_cbh_fast(arr, intensity_thr, area_thr, density_thr, min_run, min_points)
        if np.isfinite(pred):
            n_valid += 1
            errs.append(abs(pred - truth_all[k]))
    coverage = n_valid / len(keys)
    mae = np.mean(errs) if errs else np.inf
    penalty = (1 - coverage) * 10
    return mae + penalty, mae, coverage


def grid_search(keys, use_density):
    intensity_range = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    area_range = [1.1, 1.3, 1.5, 1.8, 2.0, 2.5]
    density_range = [1.0, 1.2, 1.5, 2.0] if use_density else [None]
    run_range = [2, 3, 4, 5, 6]
    points_range = [10, 20]

    results = []
    for it, ar, dr, rr, pr in itertools.product(intensity_range, area_range, density_range, run_range, points_range):
        score, mae, coverage = evaluate_fast((it, ar, dr, rr, pr), keys, use_density)
        results.append((score, mae, coverage, it, ar, dr, rr, pr))
    results.sort(key=lambda r: r[0])
    return results


all_keys = tree_keys

for use_density, label in [(False, "INTENSITY+AREA (no density)"), (True, "INTENSITY+AREA+DENSITY")]:
    print(f"\n########## {label} ##########", flush=True)
    results = grid_search(all_keys, use_density)
    print("Top 5 (full-data fit):", flush=True)
    for r in results[:5]:
        print(f"  score={r[0]:.3f} mae={r[1]:.3f} coverage={r[2]:.2f} "
              f"intensity_thr={r[3]} area_thr={r[4]} density_thr={r[5]} min_run={r[6]} min_points={r[7]}", flush=True)

    print("Leave-one-plot-out CV:", flush=True)
    for train_plot, test_plot in [("P01", "P02"), ("P02", "P01")]:
        train_keys = [k for k in all_keys if k[0] == train_plot]
        test_keys = [k for k in all_keys if k[0] == test_plot]
        train_results = grid_search(train_keys, use_density)
        best_param = train_results[0][3:]
        _, test_mae, test_coverage = evaluate_fast(best_param, test_keys, use_density)
        print(f"  train={train_plot} test={test_plot}: best_train_params={best_param} "
              f"-> test_mae={test_mae:.3f}m test_coverage={test_coverage:.2f}", flush=True)

print("\nALL DONE", flush=True)
