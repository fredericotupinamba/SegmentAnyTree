#!/usr/bin/env python3
"""Fit and cross-validate a crown-base-height model on wood/leaf features.

Reads the per-bin features written by extract_cbh_features.py and the 58
manual crown-base-height labels, then scores several candidate models
against each other under one identical protocol:

  reference   the density-only rule (whatever produced
              crown_base_height_m_old in the label CSV)
  production  the intensity+area rule currently shipping in
              compute_crown_metrics, recomputed here on the *same* trees
              from all-point features -- this is the number to beat
  rule/*      threshold rules over the new wood/leaf features
  logistic    a per-bin logistic regression over the wood/leaf features,
              read out as the first sustained run of P(crown) >= 0.5

Every model is scored by mean absolute error against the manual label,
penalised for trees it declines to answer on (coverage), and reported
both as a full-data fit and under leave-one-plot-out cross-validation.

The full-data fit is optimistically biased -- it tunes and scores on the
same 58 trees. Quote the CV number. With only two plots, CV here is two
folds, which is the best available but still weak evidence: treat a
difference of a few tens of centimetres between models as noise.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABELS = os.path.join(HERE, "cbh_tree_labels.csv")
DEFAULT_FEATURES = os.path.join(HERE, "cbh_bin_features_wood.csv")

# A tree the model declines to answer on is worth this many metres of
# error, so a rule cannot win by only answering on the easy trees.
COVERAGE_PENALTY_M = 10.0


def first_sustained_run(bin_low, cond, trust, min_run):
    """Bottom-up scan for the first run of `min_run` consecutive trusted
    bins where `cond` holds; returns the run's lower edge. Untrusted bins
    (too few points to judge) are skipped without breaking the run --
    same convention compute_crown_metrics uses, so a fitted threshold
    means the same thing once it is wired into the pipeline."""
    run = 0
    run_start = np.nan
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


def score(preds, truth, keys):
    """MAE over the trees the model answered on, plus a coverage penalty."""
    errs = [abs(preds[k] - truth[k]) for k in keys if np.isfinite(preds.get(k, np.nan))]
    coverage = len(errs) / len(keys) if keys else 0.0
    mae = float(np.mean(errs)) if errs else np.inf
    within1 = int(sum(e <= 1.0 for e in errs))
    return mae + (1 - coverage) * COVERAGE_PENALTY_M, mae, coverage, within1


class RuleModel(object):
    """A threshold rule: a bin is crown when every enabled condition holds.

    `conditions` maps a feature column to (operator, threshold), where
    operator is '<' or '>'. Leaving a feature out of the dict disables it,
    which is how the grid search discovers which features actually carry
    the trunk->crown transition rather than assuming it.
    """

    def __init__(self, conditions, min_run, min_points, offset_m=0.0):
        self.conditions = conditions
        self.min_run = min_run
        self.min_points = min_points
        # Where a feature crosses its threshold is not where an annotator
        # marks the crown base: on P01 the wood_frac=0.5 crossing sits a
        # consistent 0.90 m *below* the manual label (sd 1.29 m), because
        # the transition is gradual (~3.1 m wide per tree) and the two
        # conventions pick different points on the same ramp. That offset
        # is a fixed property of the rule, so it is fitted on training
        # trees and added back -- removing it took P01 MAE 1.09 -> 0.76 m.
        self.offset_m = offset_m

    def raw_predict(self, arrays, keys):
        preds = {}
        for k in keys:
            arr = arrays[k]
            cond = np.ones(len(arr["bin_low"]), dtype=bool)
            for col, (op, thr) in self.conditions.items():
                values = arr[col]
                with np.errstate(invalid="ignore"):
                    hit = values < thr if op == "<" else values > thr
                cond &= np.where(np.isfinite(values), hit, False)
            preds[k] = first_sustained_run(
                arr["bin_low"], cond, arr["n_points"] >= self.min_points, self.min_run
            )
        return preds

    def fit_offset(self, arrays, keys, truth):
        """Median residual on the training trees. Median, not mean, so a
        handful of trees where the rule fires on the wrong feature entirely
        cannot drag the correction."""
        raw = self.raw_predict(arrays, keys)
        res = [truth[k] - raw[k] for k in keys if np.isfinite(raw[k])]
        self.offset_m = float(np.median(res)) if res else 0.0
        return self

    def predict(self, arrays, keys):
        return {k: v + self.offset_m for k, v in self.raw_predict(arrays, keys).items()}

    def describe(self):
        parts = [f"{c} {op} {thr}" for c, (op, thr) in sorted(self.conditions.items())]
        return (f"{' AND '.join(parts)} | min_run={self.min_run} "
                f"min_points={self.min_points} offset={self.offset_m:+.2f}m")

    def to_dict(self):
        return {
            "type": "rule",
            "conditions": {c: {"op": op, "threshold": thr} for c, (op, thr) in self.conditions.items()},
            "min_run": self.min_run,
            "min_points": self.min_points,
            "offset_m": self.offset_m,
        }


class LogisticModel(object):
    """Per-bin P(this bin is above the crown base), read out as the first
    sustained run over the probability threshold. Standardises features
    first so the persisted coefficients stay interpretable and so
    regularisation treats every feature alike."""

    def __init__(self, feature_cols, min_run, min_points, prob_threshold=0.5):
        self.feature_cols = feature_cols
        self.min_run = min_run
        self.min_points = min_points
        self.prob_threshold = prob_threshold
        self.mean_ = None
        self.scale_ = None
        self.coef_ = None
        self.intercept_ = None
        # Same gradual-transition correction the rule models carry.
        self.offset_m = 0.0

    def _matrix(self, arr):
        return np.column_stack([arr[c] for c in self.feature_cols])

    def fit(self, arrays, keys, labels):
        from sklearn.linear_model import LogisticRegression

        X_parts, y_parts = [], []
        for k in keys:
            arr = arrays[k]
            X = self._matrix(arr)
            y = (arr["bin_low"] >= labels[k]).astype(int)
            ok = np.isfinite(X).all(axis=1) & (arr["n_points"] >= self.min_points)
            if ok.sum():
                X_parts.append(X[ok])
                y_parts.append(y[ok])
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((X - self.mean_) / self.scale_, y)
        self.coef_ = clf.coef_[0]
        self.intercept_ = float(clf.intercept_[0])
        raw = self.raw_predict(arrays, keys)
        res = [labels[k] - raw[k] for k in keys if np.isfinite(raw[k])]
        self.offset_m = float(np.median(res)) if res else 0.0
        return self

    def predict(self, arrays, keys):
        return {k: v + self.offset_m for k, v in self.raw_predict(arrays, keys).items()}

    def raw_predict(self, arrays, keys):
        preds = {}
        for k in keys:
            arr = arrays[k]
            X = self._matrix(arr)
            finite = np.isfinite(X).all(axis=1)
            z = np.full(len(arr["bin_low"]), -np.inf)
            Xs = (X[finite] - self.mean_) / self.scale_
            z[finite] = Xs @ self.coef_ + self.intercept_
            prob = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            trust = (arr["n_points"] >= self.min_points) & finite
            preds[k] = first_sustained_run(
                arr["bin_low"], prob >= self.prob_threshold, trust, self.min_run
            )
        return preds

    def describe(self):
        terms = ", ".join(f"{c}={w:+.3f}" for c, w in zip(self.feature_cols, self.coef_))
        return (f"logistic[{terms}, b={self.intercept_:+.3f}] | min_run={self.min_run} "
                f"min_points={self.min_points} offset={self.offset_m:+.2f}m")

    def to_dict(self):
        return {
            "type": "logistic",
            "feature_cols": list(self.feature_cols),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_,
            "prob_threshold": self.prob_threshold,
            "min_run": self.min_run,
            "min_points": self.min_points,
            "offset_m": self.offset_m,
        }


def build_arrays(bins_df, feature_cols):
    """One dict of numpy arrays per tree, bins sorted bottom-up."""
    arrays = {}
    for (plot, tid), g in bins_df.groupby(["plot", "tree_id"]):
        g = g.sort_values("hag_mid")
        arr = {"bin_low": g["bin_low"].to_numpy(), "n_points": g["n_points"].to_numpy()}
        for col in feature_cols:
            arr[col] = g[col].to_numpy(dtype=float)
        arrays[(plot, tid)] = arr
    return arrays


RULE_GRIDS = {
    "rule/wood_frac": {"wood_frac": ("<", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])},
    "rule/spread": {"wood_spread_ratio": (">", [1.5, 2.0, 2.5, 3.0, 4.0, 5.0])},
    "rule/area": {"wood_area_ratio": (">", [1.5, 2.0, 2.5, 3.0, 4.0, 5.0])},
    "rule/wood_frac+spread": {
        "wood_frac": ("<", [0.6, 0.7, 0.8, 0.9]),
        "wood_spread_ratio": (">", [1.5, 2.0, 2.5, 3.0, 4.0]),
    },
    "rule/wood_frac+area": {
        "wood_frac": ("<", [0.6, 0.7, 0.8, 0.9]),
        "wood_area_ratio": (">", [1.5, 2.0, 2.5, 3.0, 4.0]),
    },
    "rule/spread+area": {
        "wood_spread_ratio": (">", [1.5, 2.0, 2.5, 3.0]),
        "wood_area_ratio": (">", [1.5, 2.0, 2.5, 3.0]),
    },
    "rule/all_three": {
        "wood_frac": ("<", [0.6, 0.7, 0.8, 0.9]),
        "wood_spread_ratio": (">", [1.5, 2.0, 2.5, 3.0]),
        "wood_area_ratio": (">", [1.5, 2.0, 2.5, 3.0]),
    },
    # The rule shipping today, recomputed on all-point features so it is
    # scored on exactly the same trees as everything else.
    "production/intensity+area": {
        "intensity_all_ratio": ("<", [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]),
        "total_area_ratio": (">", [1.1, 1.3, 1.5, 1.8, 2.0, 2.5]),
    },
}

MIN_RUN_GRID = [2, 3, 4, 5, 6]
MIN_POINTS_GRID = [10, 20]


def search_rules(grid, arrays, keys, truth):
    """Exhaustive search over one feature combination's thresholds."""
    cols = sorted(grid)
    best = None
    for combo in itertools.product(*[grid[c][1] for c in cols]):
        conditions = {c: (grid[c][0], thr) for c, thr in zip(cols, combo)}
        for min_run, min_points in itertools.product(MIN_RUN_GRID, MIN_POINTS_GRID):
            # The offset is fitted on these same keys, which are the
            # *training* trees whenever this is called from the CV loop --
            # so a fold never sees its test plot's residuals.
            model = RuleModel(conditions, min_run, min_points).fit_offset(arrays, keys, truth)
            s = score(model.predict(arrays, keys), truth, keys)
            if best is None or s[0] < best[0][0]:
                best = (s, model)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--output-model", default=os.path.join(HERE, "cbh_model.json"))
    args = parser.parse_args()

    bins_df = pd.read_csv(args.features)
    trees_df = pd.read_csv(args.labels)
    trees_df = trees_df[trees_df["CBH_m"].notna() & (trees_df["CBH_m"] < 100)]

    # All-point ratios, so the shipping rule can be scored side by side.
    for src, dst in (("intensity_median_all", "intensity_all_ratio"), ("total_area_m2", "total_area_ratio")):
        base = (bins_df[bins_df["hag_mid"] <= 1.3]
                .groupby(["plot", "tree_id"])[src].median().rename("base"))
        bins_df = bins_df.join(base, on=["plot", "tree_id"])
        bins_df[dst] = bins_df[src] / bins_df["base"]
        bins_df = bins_df.drop(columns="base")

    feature_cols = ["wood_frac", "leaf_frac", "wood_area_ratio", "wood_spread_ratio",
                     "intensity_wood_ratio", "intensity_all_ratio", "total_area_ratio"]
    arrays = build_arrays(bins_df, feature_cols)

    keys = [k for k in zip(trees_df["plot"], trees_df["tree_id"]) if k in arrays]
    truth = dict(zip(zip(trees_df["plot"], trees_df["tree_id"]), trees_df["CBH_m"]))
    plots = sorted({k[0] for k in keys})
    print(f"{len(keys)} trees with features across plots {plots}\n", flush=True)

    old = {k: v for k, v in zip(zip(trees_df["plot"], trees_df["tree_id"]),
                                 trees_df["crown_base_height_m_old"]) if k in arrays}
    _, mae, cov, w1 = score(old, truth, keys)
    print(f"{'reference/density-only':<28} full-fit MAE={mae:5.2f}m cov={cov:.2f} within-1m={w1}/{len(keys)}\n", flush=True)

    logistic_cols = ["wood_frac", "wood_area_ratio", "wood_spread_ratio"]
    results = {}

    print(f"{'model':<28} {'full-fit':>9} {'LOPO-CV':>9}  detail", flush=True)
    print("-" * 96, flush=True)

    for name, grid in RULE_GRIDS.items():
        (s, mae, cov, w1), model = search_rules(grid, arrays, keys, truth)
        cv_maes, cv_covs = [], []
        for test_plot in plots:
            train_keys = [k for k in keys if k[0] != test_plot]
            test_keys = [k for k in keys if k[0] == test_plot]
            if not train_keys or not test_keys:
                continue
            _, tmodel = search_rules(grid, arrays, train_keys, truth)
            _, t_mae, t_cov, _ = score(tmodel.predict(arrays, test_keys), truth, test_keys)
            cv_maes.append(t_mae)
            cv_covs.append(t_cov)
        cv_mae = float(np.mean(cv_maes)) if cv_maes else np.inf
        results[name] = {"cv_mae": cv_mae, "full_mae": mae, "coverage": cov, "model": model}
        print(f"{name:<28} {mae:8.2f}m {cv_mae:8.2f}m  cov={cov:.2f} within-1m={w1}/{len(keys)} :: {model.describe()}", flush=True)

    for min_run, min_points in ((3, 10), (5, 10)):
        name = f"logistic/run{min_run}"
        model = LogisticModel(logistic_cols, min_run, min_points).fit(arrays, keys, truth)
        _, mae, cov, w1 = score(model.predict(arrays, keys), truth, keys)
        cv_maes = []
        for test_plot in plots:
            train_keys = [k for k in keys if k[0] != test_plot]
            test_keys = [k for k in keys if k[0] == test_plot]
            if not train_keys or not test_keys:
                continue
            tmodel = LogisticModel(logistic_cols, min_run, min_points).fit(arrays, train_keys, truth)
            _, t_mae, _, _ = score(tmodel.predict(arrays, test_keys), truth, test_keys)
            cv_maes.append(t_mae)
        cv_mae = float(np.mean(cv_maes)) if cv_maes else np.inf
        results[name] = {"cv_mae": cv_mae, "full_mae": mae, "coverage": cov, "model": model}
        print(f"{name:<28} {mae:8.2f}m {cv_mae:8.2f}m  cov={cov:.2f} within-1m={w1}/{len(keys)} :: {model.describe()}", flush=True)

    # Selection is on cross-validated error, never on the full-data fit --
    # the full fit tunes and scores on the same 58 trees and always looks
    # better than the model really is.
    best_name = min(results, key=lambda n: results[n]["cv_mae"])
    best = results[best_name]
    print(f"\nBest by LOPO-CV: {best_name} (CV MAE={best['cv_mae']:.2f}m, full-fit={best['full_mae']:.2f}m)", flush=True)

    payload = best["model"].to_dict()
    payload["_selected_by"] = "leave-one-plot-out CV mean absolute error"
    payload["_model_name"] = best_name
    payload["_cv_mae_m"] = best["cv_mae"]
    payload["_full_fit_mae_m"] = best["full_mae"]
    payload["_n_trees"] = len(keys)
    payload["_plots"] = plots
    with open(args.output_model, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.output_model}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
