"""Test solvent-accessible surface area methods."""
import logging
import pytest
import numpy as np
from osmolytes.sasa import SolventAccessibleSurface
from osmolytes.pqr import parse_pqr_file


_LOGGER = logging.getLogger(__name__)

@pytest.mark.parametrize("radius", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_sphere_sasa(radius):
    """Test basic functions of SolventAccessible Surface class."""
    with open("tests/data/sphere.pqr") as pqr_file:
        atoms = parse_pqr_file(pqr_file)
    atoms[0].position = np.random.randn(3)
    frac = np.random.rand(1)[0]
    atoms[0].radius = frac*radius
    probe_radius = (1.0-frac)*radius
    sas = SolventAccessibleSurface(atoms, probe_radius, 500)
    atom_sasa = sas.atom_surface_area(0)
    ref_sasa = 4.0 * np.pi * radius * radius
    np.testing.assert_almost_equal(atom_sasa, ref_sasa)
