import numpy as np
import pandas as pd

from tupisat_inference.split_merged_instances import split_merged_instances


def _cylinder_points(xc, yc, radius_m, z_top, z_bottom=0.0, n_rings=40, points_per_ring=60, seed=0):
    rng = np.random.default_rng(seed)
    zs = np.linspace(z_bottom, z_top, n_rings)
    rows = []
    for z in zs:
        theta = rng.uniform(0, 2 * np.pi, points_per_ring)
        x = xc + radius_m * np.cos(theta)
        y = yc + radius_m * np.sin(theta)
        rows.append(np.column_stack([x, y, np.full(points_per_ring, z)]))
    return np.vstack(rows)


def test_splits_two_merged_stems_into_separate_instances():
    stem_a = _cylinder_points(xc=0.0, yc=0.0, radius_m=0.15, z_top=12.0, seed=1)
    stem_b = _cylinder_points(xc=3.6, yc=0.0, radius_m=0.15, z_top=12.0, seed=2)
    xyz = np.vstack([stem_a, stem_b])
    df = pd.DataFrame(xyz, columns=["x", "y", "z"])
    df["PredSemantic"] = 1
    df["PredInstance"] = 5

    result = split_merged_instances(df.copy())

    ids = result["PredInstance"].unique()
    assert len(ids) == 2

    # Each resulting instance should be spatially coherent -- close to one
    # of the two original stem centers, not a mix of both.
    for stem_id in ids:
        pts = result.loc[result["PredInstance"] == stem_id, ["x", "y"]].to_numpy()
        centroid = pts.mean(axis=0)
        dist_to_a = np.hypot(centroid[0] - 0.0, centroid[1] - 0.0)
        dist_to_b = np.hypot(centroid[0] - 3.6, centroid[1] - 0.0)
        assert min(dist_to_a, dist_to_b) < 0.3


def test_single_stem_is_not_split():
    stem = _cylinder_points(xc=0.0, yc=0.0, radius_m=0.15, z_top=12.0, seed=3)
    df = pd.DataFrame(stem, columns=["x", "y", "z"])
    df["PredSemantic"] = 1
    df["PredInstance"] = 7

    result = split_merged_instances(df.copy())

    assert (result["PredInstance"] == 7).all()


def test_non_tree_and_unassigned_points_are_untouched():
    stem_a = _cylinder_points(xc=0.0, yc=0.0, radius_m=0.15, z_top=12.0, seed=1)
    stem_b = _cylinder_points(xc=3.6, yc=0.0, radius_m=0.15, z_top=12.0, seed=2)
    tree_df = pd.DataFrame(np.vstack([stem_a, stem_b]), columns=["x", "y", "z"])
    tree_df["PredSemantic"] = 1
    tree_df["PredInstance"] = 9

    ground_df = pd.DataFrame(
        {"x": [10.0, 11.0], "y": [10.0, 11.0], "z": [0.0, 0.0], "PredSemantic": [0, 0], "PredInstance": [0, 0]}
    )
    unassigned_df = pd.DataFrame(
        {"x": [20.0, 21.0], "y": [20.0, 21.0], "z": [0.0, 0.0], "PredSemantic": [1, 1], "PredInstance": [0, 0]}
    )
    df = pd.concat([tree_df, ground_df, unassigned_df], ignore_index=True)

    result = split_merged_instances(df.copy())

    ground_result = result.iloc[len(tree_df) : len(tree_df) + 2]
    unassigned_result = result.iloc[len(tree_df) + 2 :]
    assert (ground_result["PredInstance"] == 0).all()
    assert (unassigned_result["PredInstance"] == 0).all()
    assert result.loc[: len(tree_df) - 1, "PredInstance"].nunique() == 2


def test_two_separate_single_stem_instances_stay_separate_and_unsplit():
    stem_a = _cylinder_points(xc=0.0, yc=0.0, radius_m=0.15, z_top=12.0, seed=1)
    stem_b = _cylinder_points(xc=20.0, yc=20.0, radius_m=0.15, z_top=12.0, seed=2)
    df_a = pd.DataFrame(stem_a, columns=["x", "y", "z"])
    df_a["PredSemantic"] = 1
    df_a["PredInstance"] = 1
    df_b = pd.DataFrame(stem_b, columns=["x", "y", "z"])
    df_b["PredSemantic"] = 1
    df_b["PredInstance"] = 2
    df = pd.concat([df_a, df_b], ignore_index=True)

    result = split_merged_instances(df.copy())

    assert set(result["PredInstance"].unique()) == {1, 2}
    assert (result.loc[: len(df_a) - 1, "PredInstance"] == 1).all()
    assert (result.loc[len(df_a) :, "PredInstance"] == 2).all()
