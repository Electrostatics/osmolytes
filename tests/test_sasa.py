"""Test solvent-accessible surface area methods."""
import logging
import json
from pathlib import Path
from random import uniform
import pytest
import numpy as np
import pandas as pd
from osmolytes.sasa import SolventAccessibleSurface
from osmolytes.pqr import parse_pqr_file, Atom


_LOGGER = logging.getLogger(__name__)


with open("tests/data/alkanes/alkanes.json", "rt") as json_file:
    ATOM_AREAS = json.load(json_file)


@pytest.mark.parametrize("radius", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_one_sphere_sasa(radius, tmp_path):
    """Test solvent-accessible surface areas for one sphere."""
    atom = Atom()
    atom.position = np.random.randn(3)
    frac = np.random.rand(1)[0]
    atom.radius = frac * radius
    probe_radius = (1.0 - frac) * radius
    xyz_path = Path(tmp_path) / "sphere.xyz"
    sas = SolventAccessibleSurface(
        [atom], probe_radius, 1000, xyz_path=xyz_path
    )
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


@pytest.mark.parametrize("radius", [0.0, 1.1, 2.2, 4.4, 6.6, 8.8])
def test_two_sphere_sasa(radius, tmp_path):
    """Test solvent accessible surface areas for two spheres."""
    atom_tolerance = 0.02
    total_tolerance = 0.02
    probe_radius = 0.0
    big_atom = Atom()
    big_atom.radius = radius
    big_atom.position = np.array([0, 0, 0])
    little_atom = Atom()
    little_atom.radius = 1.0
    test_atom_areas = []
    test_total_areas = []
    ref_atom_areas = []
    ref_total_areas = []
    distances = np.linspace(0, (big_atom.radius + little_atom.radius), num=20)
    for distance in distances:
        _LOGGER.debug("Distance = %g", distance)
        little_atom.position = np.array(3 * [distance / np.sqrt(3)])
        xyz_path = Path(tmp_path) / f"spheres-{distance}.xyz"
        sas = SolventAccessibleSurface(
            [big_atom, little_atom], probe_radius, 1000, xyz_path=xyz_path
        )
        test = np.array([sas.atom_surface_area(0), sas.atom_surface_area(1)])
        test_total_areas.append(test.sum())
        test_atom_areas.append(test)
        ref = np.array(
            two_sphere_area(big_atom.radius, little_atom.radius, distance)
        )
        ref_total_areas.append(ref.sum())
        ref_atom_areas.append(ref)
    test_atom_areas = np.array(test_atom_areas)
    test_total_areas = np.array(test_total_areas)
    ref_atom_areas = np.array(ref_atom_areas)
    ref_total_areas = np.array(ref_total_areas)
    rel_difference = np.absolute(
        np.divide(test_atom_areas - ref_atom_areas, np.sum(ref_atom_areas))
    )
    errors = []
    if np.any(rel_difference > atom_tolerance):
        ref_series = pd.Series(index=distances, data=ref_total_areas)
        ref_series.index.name = "Dist"
        ref_df = pd.DataFrame(index=distances, data=ref_atom_areas)
        ref_df.index.name = "Dist"
        test_df = pd.DataFrame(index=distances, data=test_atom_areas)
        test_df.index.name = "Dist"
        abs_diff_df = pd.DataFrame(
            index=distances,
            data=np.absolute(ref_atom_areas - test_atom_areas),
        )
        abs_diff_df.index.name = "Dist"
        rel_diff_df = pd.DataFrame(index=distances, data=rel_difference)
        rel_diff_df.index.name = "Dist"
        _LOGGER.error(
            f"\nTolerance exceeded: {atom_tolerance}\n"
            f"\nRadii: {little_atom.radius}, {big_atom.radius}\n"
            f"\nReference total areas:\n{ref_series}\n"
            f"\nReference atom areas:\n{ref_df}\n"
            f"\nTest atom areas:\n{test_df}\n"
            f"\nAbsolute differences:\n{abs_diff_df}\n"
            f"\nRelative differences:\n{rel_diff_df}\n"
        )
        errors.append(
            "Atomic surface area relative differences exceed tolerance (%g)"
            % atom_tolerance
        )
    rel_differences = np.absolute(
        np.divide(test_total_areas - ref_total_areas, ref_total_areas)
    )
    if np.any(rel_differences > total_tolerance):
        ref_series = pd.Series(index=distances, data=ref_total_areas)
        ref_series.index.name = "Dist"
        test_series = pd.Series(index=distances, data=test_total_areas)
        test_series.index.name = "Dist"
        abs_series = pd.Series(
            index=distances,
            data=np.absolute(ref_total_areas - test_total_areas),
        )
        abs_series.index.name = "Dist"
        rel_series = pd.Series(index=distances, data=rel_differences)
        rel_series.index.name = "Dist"
        _LOGGER.error(
            f"\nTolerance exceeded:  {total_tolerance}\n"
            f"\nRadii: {little_atom.radius}, {big_atom.radius}\n"
            f"\nReference areas:\n{ref_series}\n"
            f"\nTest areas:\n{test_series}\n"
            f"\nAbsolute differences:\n{abs_series}\n"
            f"\nRelative differences:\n{rel_series}"
        )
        errors.append(
            "Total surface area relative differences exceed tolerance (%g)"
            % total_tolerance
        )
    if errors:
        raise AssertionError(";".join(errors))


@pytest.mark.parametrize("molecule", list(ATOM_AREAS.keys()))
def test_atom_sasa(molecule, tmp_path):
    """Test per-atom solvent-accessible surface areas for small molecules."""
    atom_tolerance = 0.03
    total_tolerance = 0.03
    pqr_path = "%s.pdb" % molecule
    pqr_path = Path("tests/data/alkanes") / pqr_path
    with open(pqr_path, "rt") as pqr_file:
        atoms = parse_pqr_file(pqr_file)
    xyz_path = Path(tmp_path) / f"{molecule}.xyz"
    sas = SolventAccessibleSurface(atoms, 0.65, 1000, xyz_path=xyz_path)
    test_areas = np.array(
        [sas.atom_surface_area(iatom) for iatom in range(len(atoms))]
    )
    test_areas = test_areas
    ref_areas = np.array(ATOM_AREAS[molecule])
    ref_areas = ref_areas
    abs_diff = np.absolute(test_areas - ref_areas)
    rel_diff = abs_diff / np.sum(ref_areas)
    errors = []
    if np.any(rel_diff > atom_tolerance):
        _LOGGER.warning(
            f"Tolerance exceeded: {atom_tolerance}\n"
            f"Reference areas:\n{ref_areas}\n"
            f"Test areas:\n{ref_areas}\n"
            f"Absolute differences:\n{abs_diff}\n"
            f"Relative differences:\n{rel_diff}\n"
        )
        errors.append(
            "Per-atom relative differences exceed tolerance (%g)"
            % atom_tolerance
        )
    abs_error = np.absolute(np.sum(test_areas) - np.sum(ref_areas))
    rel_error = abs_error / np.sum(ref_areas)
    if rel_error > total_tolerance:
        _LOGGER.warning(
            f"Tolerance exceeded: {total_tolerance}\n"
            f"Reference area: {np.sum(ref_areas)}\n"
            f"Test area: {np.sum(test_areas)}\n"
            f"Absolute difference: {abs_error}\n"
            f"Relative difference: {rel_error}\n"
        )
        errors.append(
            "Total relative difference exceeds tolerance (%g)"
            % total_tolerance
        )
    if errors:
        raise AssertionError(";".join(errors))
