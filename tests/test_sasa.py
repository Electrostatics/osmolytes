"""Test solvent-accessible surface area methods."""
import logging
import pytest
from pathlib import Path
from random import uniform
import numpy as np
from osmolytes.sasa import SolventAccessibleSurface
from osmolytes.pqr import parse_pqr_file, Atom


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
def test_one_sphere_sasa(radius):
    """Test solvent-accessible surface areas for one sphere."""
    atom = Atom()
    atom.position = np.random.randn(3)
    frac = np.random.rand(1)[0]
    atom.radius = frac * radius
    probe_radius = (1.0 - frac) * radius
    sas = SolventAccessibleSurface([atom], probe_radius, 500)
    atom_sasa = sas.atom_surface_area(0)
    ref_sasa = 4.0 * np.pi * radius * radius
    np.testing.assert_almost_equal(atom_sasa, ref_sasa)


def two_sphere_area(radius1, radius2, distance):
    """Area of two overlapping spheres.

    :param float radius1:  radius of sphere1
    :param float radius2:  radius of sphere2
    :param float distance:  distance between centers of spheres
    :returns:  exposed areas of spheres
    :rtype:  (float, float)
    """
    distsq = distance * distance
    rad1sq = radius1 * radius1
    rad2sq = radius2 * radius2
    full_area1 = 4 * np.pi * rad1sq
    full_area2 = 4 * np.pi * rad2sq
    if distance > (radius1 + radius2):
        return (full_area1, full_area2)
    elif distance <= np.absolute(radius1 - radius2):
        if full_area1 > full_area2:
            return (full_area1, 0)
        if full_area1 < full_area2:
            return (0, full_area2)
        else:
            return (0.5 * full_area1, 0.5 * full_area2)
    else:
        if radius1 > 0:
            cos_theta1 = (rad1sq + distsq - rad2sq) / (2 * radius1 * distance)
            cap1_area = 2 * np.pi * radius1 * radius1 * (1 - cos_theta1)
        else:
            cap1_area = 0
        if radius2 > 0:
            cos_theta2 = (rad2sq + distsq - rad1sq) / (2 * radius2 * distance)
            cap2_area = 2 * np.pi * radius2 * radius2 * (1 - cos_theta2)
        else:
            cap2_area = 0
        return (full_area1 - cap1_area, full_area2 - cap2_area)


@pytest.mark.parametrize("radius", [0, 1, 2, 4, 6, 8])
def test_two_sphere_sasa(radius):
    """Test solvent accessible surface areas for two spheres."""
    tolerance = 0.05
    probe_radius = 0.0
    big_atom = Atom()
    big_atom.radius = radius
    big_atom.position = np.array([0, 0, 0])
    little_atom = Atom()
    little_atom.radius = 1.0
    test_data = []
    ref_data = []
    total_areas = []
    distances = np.linspace(
        np.absolute(big_atom.radius - little_atom.radius),
        (big_atom.radius + little_atom.radius),
        num=10,
    )
    for distance in distances:
        little_atom.position = np.array(3 * [distance / np.sqrt(3)])
        sas = SolventAccessibleSurface(
            [big_atom, little_atom], probe_radius, 5000
        )
        test = np.array([sas.atom_surface_area(0), sas.atom_surface_area(1)])
        test = test / test.sum()
        test_data.append(test)
        ref = np.array(
            two_sphere_area(big_atom.radius, little_atom.radius, distance)
        )
        total_areas.append(ref.sum())
        ref = ref / ref.sum()
        ref_data.append(ref)
    ref_data = np.array(ref_data)
    test_data = np.array(test_data)
    total_areas = np.array(total_areas)
    if np.any(test_data - ref_data > tolerance):
        _LOGGER.error(
            f"Tolerance exceeded: {tolerance}\n"
            f"Distances:\n{distances}\n"
            f"Areas:\n{total_areas}\n"
            f"Differences:\n{ref_data - test_data}"
            f"Reference values:\n{ref_data}"
        )
        raise AssertionError("Differences exceed tolerance (%g)" % tolerance)


@pytest.mark.parametrize("molecule", ["ethane", "methane", "butane"])
def test_atom_sasa(molecule):
    """Test per-atom solvent-accessible surface areas for small molecules."""
    tolerance = 0.02
    pqr_path = "%s.pdb" % molecule
    pqr_path = Path("tests/data/alkanes") / pqr_path
    with open(pqr_path, "rt") as pqr_file:
        atoms = parse_pqr_file(pqr_file)
    sas = SolventAccessibleSurface(atoms, 0.65, 900)
    test_areas = np.array(
        [sas.atom_surface_area(iatom) for iatom in range(len(atoms))]
    )
    test_areas = test_areas / np.sum(test_areas)
    ref_areas = np.array(ATOM_AREAS[molecule])
    total_areas = ref_areas
    ref_areas = ref_areas / np.sum(ref_areas)
    if np.any(np.absolute(test_areas - ref_areas) > tolerance):
        _LOGGER.warning(
            f"Tolerance exceeded: {tolerance}\n"
            f"Areas:\n{total_areas}\n"
            f"Differences:\n{ref_areas - test_areas}\n"
        )
        raise AssertionError("Differences exceed tolerance (%g)" % tolerance)
