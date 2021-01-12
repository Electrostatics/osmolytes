"""Test SASA reference sphere code."""
import numpy as np
from numpy.testing import assert_almost_equal
from osmolytes.sasa import sphere


def test_sphere():
    """Test SASA reference sphere code."""
    num_points = 2000
    test_points = sphere(num_points)
    ref_points = num_points * [1.0]
    assert_almost_equal(np.linalg.norm(test_points, axis=1), ref_points)
