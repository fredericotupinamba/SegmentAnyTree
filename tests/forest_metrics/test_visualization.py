import numpy as np

from tupisat_inference.forest_metrics.visualization import (
    circle_points,
    cross_marker_points,
    render_text_points,
)


def test_circle_points_lie_on_the_circle():
    pts = circle_points(xc=10.0, yc=-5.0, z=100.0, radius_m=0.2, n_points=16)

    assert pts.shape == (16, 3)
    dist = np.hypot(pts[:, 0] - 10.0, pts[:, 1] - (-5.0))
    assert np.allclose(dist, 0.2, atol=1e-9)
    assert np.all(pts[:, 2] == 100.0)


def test_circle_points_invalid_radius_returns_empty():
    assert circle_points(0, 0, 0, radius_m=0, n_points=16).shape[0] == 0
    assert circle_points(0, 0, 0, radius_m=np.nan, n_points=16).shape[0] == 0


def test_cross_marker_points_centered_on_target():
    pts = cross_marker_points(x=5.0, y=5.0, z=5.0, size_m=0.1, points_per_arm=5)

    assert pts.shape == (15, 3)
    assert np.allclose(pts.mean(axis=0), [5.0, 5.0, 5.0], atol=1e-9)


def test_cross_marker_points_nan_input_returns_empty():
    assert cross_marker_points(np.nan, 0, 0, size_m=0.1).shape[0] == 0


def test_render_text_points_nonempty_for_real_text():
    pts = render_text_points("ID:5", character_size=0.05, x=0.0, y=0.0, z=0.0)
    assert pts.shape[0] > 0
    assert pts.shape[1] == 3


def test_render_text_points_empty_string_returns_empty():
    pts = render_text_points("", character_size=0.05, x=0.0, y=0.0, z=0.0)
    assert pts.shape[0] == 0


def test_render_text_points_unsupported_character_does_not_raise():
    pts = render_text_points("ID#5", character_size=0.05, x=0.0, y=0.0, z=0.0)
    assert pts.shape[1] == 3
