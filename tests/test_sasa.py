"""Test solvent-accessible surface area methods."""
import logging
import pytest
from pathlib import Path
import numpy as np
from osmolytes.sasa import SolventAccessibleSurface
from osmolytes.pqr import parse_pqr_file


_LOGGER = logging.getLogger(__name__)
ATOM_AREAS = {
    "methane": [
        1.231304303117e01,
        2.323233009850e01,
        2.345641941977e01,
        2.377431357320e01,
        2.264344420771e01,
    ],
    "ethane": [
        5.995981536705e00,
        5.966113633657e00,
        2.121552620704e01,
        2.124158310486e01,
        2.125200586399e01,
        2.123116034573e01,
        2.125200586399e01,
        2.127285138225e01,
    ],
    "butane": [
        4.405515699447e00,
        8.213673337951e-01,
        8.064333822716e-01,
        4.375647796400e00,
        1.855251124959e01,
        2.147609518526e01,
        1.852645435176e01,
        1.660345529247e01,
        1.658782115377e01,
        1.658260977421e01,
        1.658260977421e01,
        2.145003828744e01,
        1.852124297220e01,
        1.856293400871e01,
    ],
}


@pytest.mark.parametrize("radius", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_sphere_sasa(radius):
    """Test solvent-accessible surface areas for spheres."""
    with open("tests/data/sphere.pqr", "rt") as pqr_file:
        atoms = parse_pqr_file(pqr_file)
    atoms[0].position = np.random.randn(3)
    frac = np.random.rand(1)[0]
    atoms[0].radius = frac * radius
    probe_radius = (1.0 - frac) * radius
    sas = SolventAccessibleSurface(atoms, probe_radius, 500)
    atom_sasa = sas.atom_surface_area(0)
    ref_sasa = 4.0 * np.pi * radius * radius
    np.testing.assert_almost_equal(atom_sasa, ref_sasa)


@pytest.mark.parametrize("molecule", ["ethane", "methane", "butane"])
def test_atom_sasa(molecule):
    """Test per-atom solvent-accessible surface areas for small molecules."""
    pqr_path = "%s.pdb" % molecule
    pqr_path = Path("tests/data/alkanes") / pqr_path
    with open(pqr_path, "rt") as pqr_file:
        atoms = parse_pqr_file(pqr_file)
    sas = SolventAccessibleSurface(atoms, 0.65, 900)
    test_areas = [sas.atom_surface_area(iatom) for iatom in range(len(atoms))]
    ref_areas = ATOM_AREAS[molecule]
    np.testing.assert_almost_equal(test_areas, ref_areas)
