"""Test SASA reference sphere code."""
import numpy as np
from numpy.testing import assert_almost_equal
from osmolytes.sasa import sphere


REF_POINTS = np.array(
    [
        [-12.000, 0.000, 0.000],
        [12.000, 0.000, 0.000],
        [0.000, -12.000, 0.000],
        [0.000, 12.000, 0.000],
        [0.000, 0.000, -12.000],
        [0.000, 0.000, 12.000],
        [-6.928, -6.928, -6.928],
        [-6.928, -6.928, 6.928],
        [-6.928, 6.928, -6.928],
        [-6.928, 6.928, 6.928],
        [6.928, -6.928, -6.928],
        [6.928, -6.928, 6.928],
        [6.928, 6.928, -6.928],
        [6.928, 6.928, 6.928],
        [-8.485, -8.485, 0.000],
        [-8.485, 0.000, -8.485],
        [0.000, -8.485, -8.485],
        [-8.485, 0.000, 8.485],
        [0.000, -8.485, 8.485],
        [-8.485, 8.485, 0.000],
        [-8.485, 0.000, -8.485],
        [0.000, 8.485, -8.485],
        [-8.485, 0.000, 8.485],
        [0.000, 8.485, 8.485],
        [8.485, -8.485, 0.000],
        [8.485, 0.000, -8.485],
        [0.000, -8.485, -8.485],
        [8.485, 0.000, 8.485],
        [0.000, -8.485, 8.485],
        [8.485, 8.485, 0.000],
        [8.485, 0.000, -8.485],
        [0.000, 8.485, -8.485],
        [8.485, 0.000, 8.485],
        [0.000, 8.485, 8.485],
    ]
)


def test_sphere():
    """Test SASA reference sphere code."""
    test_points = 0.3*40*sphere(num=40)
    assert_almost_equal(test_points, REF_POINTS, decimal=3)
