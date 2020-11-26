"""Test solvent-accessible surface area methods."""
import logging
import json
from pathlib import Path
import pytest
import yaml
from scipy.stats import linregress
import numpy as np
import pandas as pd
from osmolytes.sasa import SolventAccessibleSurface, ReferenceModels
from osmolytes.pqr import parse_pqr_file, Atom, aggregate, count_residues


_LOGGER = logging.getLogger(__name__)


with open("tests/data/alkanes/alkanes.json", "rt") as json_file:
    ATOM_AREAS = json.load(json_file)
PROTEIN_PATH = Path("tests/data/proteins")


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
    _LOGGER.info("Temp path:  %s", tmp_path)
    atom_tolerance = 0.03
    total_tolerance = 0.03
    pqr_path = "%s.pdb" % molecule
    pqr_path = Path("tests/data/alkanes") / pqr_path
    with open(pqr_path, "rt") as pqr_file:
        atoms = parse_pqr_file(pqr_file)
    xyz_path = Path(tmp_path) / f"{molecule}.xyz"
    sas = SolventAccessibleSurface(atoms, 0.65, 4574, xyz_path=xyz_path)
    test_areas = np.array(
        [sas.atom_surface_area(iatom) for iatom in range(len(atoms))]
    )
    test_areas = test_areas
    ref_areas = np.array(ATOM_AREAS[molecule])
    ref_areas = ref_areas
    abs_diff = np.absolute(test_areas - ref_areas)
    rel_diff = abs_diff / np.sum(ref_areas)
    keys = []
    for atom in atoms:
        keys.append(f"{atom.pqr_atom_num} {atom.atom_name}")
    df = pd.DataFrame(
        index=keys,
        data={
            "Ref values": ref_areas,
            "Test values": test_areas,
            "Abs diff": abs_diff,
            "Rel diff": rel_diff,
        },
    )
    _LOGGER.info("\n%s", df.sort_values("Abs diff", ascending=False))
    errors = []
    if np.any(rel_diff > atom_tolerance):
        _LOGGER.error("\n%s", df.sort_values("Rel diff", ascending=False))
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


@pytest.mark.parametrize(
    "pqr_path,ref_json", [("NikR_chains.pqr", "NikR.json")]
)
def test_protein_details(pqr_path, ref_json, tmp_path):
    """Test SASA performance for proteins."""
    _LOGGER.info("Temp path:  %s", tmp_path)
    abs_cutoff = 2.5
    num_abs_cutoff = 26
    rel_cutoff = 0.1
    num_rel_cutoff = 33
    tot_abs_cutoff = 24.0
    tot_rel_cutoff = 0.002
    pqr_path = PROTEIN_PATH / pqr_path
    ref_json = PROTEIN_PATH / ref_json
    with open(pqr_path, "rt") as pqr_file:
        atoms = parse_pqr_file(pqr_file)
    xyz_path = Path(tmp_path) / "surface.xyz"
    sas = SolventAccessibleSurface(
        atoms, probe_radius=1.4, num_points=5026, xyz_path=xyz_path
    )
    test_areas = [sas.atom_surface_area(iatom) for iatom in range(len(atoms))]
    df = aggregate(
        atoms,
        test_areas,
        chain_id=True,
        res_name=False,
        res_num=True,
        sidechain=False,
    )
    with open(ref_json, "rt") as ref_file:
        ref_dict = json.load(ref_file)
    keys = []
    ref_values = []
    test_values = []
    for chain in ref_dict.keys():
        chain_df = df[df["chain_id"] == chain]
        test_values = test_values + list(chain_df["value"].values)
        keys = keys + [f"{chain} {key}" for key in ref_dict[chain].keys()]
        ref_values = ref_values + list(ref_dict[chain].values())
    ref_values = np.array(ref_values)
    test_values = np.array(test_values)
    abs_diff = np.absolute(ref_values - test_values)
    rel_diff = np.divide(abs_diff, ref_values)
    df = pd.DataFrame(
        index=keys,
        data={
            "Ref values": ref_values,
            "Test values": test_values,
            "Abs diff": abs_diff,
            "Rel diff": rel_diff,
        },
    )
    _LOGGER.info(
        "Largest absolute differences:\n%s", df.nlargest(20, "Abs diff")
    )
    df = df.dropna()
    _LOGGER.info(
        "Largest relative differences:\n%s", df.nlargest(20, "Rel diff")
    )
    errors = []
    num_abs = len(abs_diff[abs_diff > abs_cutoff])
    if num_abs > num_abs_cutoff:
        err = (
            f"Number of absolute differences ({num_abs}) above {abs_cutoff} "
            f"exceeds limit ({num_abs_cutoff})"
        )
        errors.append(err)
    num_rel = len(rel_diff[rel_diff > rel_cutoff])
    if num_rel > num_rel_cutoff:
        err = (
            f"Number of relative differences ({num_rel}) above {rel_cutoff}) "
            f"exceeds limit ({num_rel_cutoff})"
        )
        errors.append(err)
    ref_total = np.sum(ref_values)
    test_total = np.sum(test_values)
    abs_total = np.absolute(ref_total - test_total)
    if abs_total > tot_abs_cutoff:
        err = (
            f"Absolute difference in total area ({abs_total}) exceeds "
            f"limit ({tot_abs_cutoff})"
        )
        errors.append(err)
    rel_total = abs_total / ref_total
    if rel_total > tot_rel_cutoff:
        err = (
            f"Relative difference in total area ({rel_total}) exceeds "
            "limit ({tot_rel_cutoff})"
        )
        errors.append(err)
    if errors:
        raise AssertionError(";".join(errors))


@pytest.mark.parametrize("protein", ["1A6F", "1STN", "2BU4"])
def test_protein_aggregate(protein, tmp_path):
    """Test aggregate SASAs per residue type as reported in Auton and Bolen
    (doi:10.1073/pnas.0507053102, Supporting Table 2)."""
    _LOGGER.info("Temp path:  %s", tmp_path)
    pqr_path = PROTEIN_PATH / f"{protein}.pqr"
    with open(pqr_path, "rt") as pqr_file:
        atoms = parse_pqr_file(pqr_file)
    xyz_path = Path(tmp_path) / "surface.xyz"
    sas = SolventAccessibleSurface(
        atoms, probe_radius=1.4, num_points=5000, xyz_path=xyz_path
    )
    test_areas = [sas.atom_surface_area(iatom) for iatom in range(len(atoms))]
    test_areas = aggregate(
        atoms,
        test_areas,
        chain_id=False,
        res_name=True,
        res_num=False,
        sidechain=True,
    )
    test_areas = test_areas.set_index("res_name").sort_index()
    test_counts = np.array(count_residues(atoms))
    ref_path = PROTEIN_PATH / "Auton-Bolen.yaml"
    with open(ref_path, "rt") as ref_file:
        ref_dict = yaml.load(ref_file, Loader=yaml.FullLoader)
    ref_dict = ref_dict[protein]
    ref_df = pd.DataFrame(ref_dict).T
    ref_df = ref_df[ref_df["number"] > 0]
    ref_counts = np.array(ref_df["number"].sort_index())
    np.testing.assert_equal(test_counts, ref_counts)
    for group in ["backbone", "sidechain"]:
        ref = np.array(ref_df[group])
        test = np.array(test_areas[test_areas["sidechain"] == group]["value"])
        # Test correlation since we're not 100% sure what parameters were used
        # in PNAS paper
        results = linregress(ref, test)
        info = (
            f"{group} areas are correlated with results from paper:  "
            f"{results}"
        )
        _LOGGER.info(info)
        if not np.isclose(results.pvalue, 0):
            err = f"Poor fit to {group} areas:  {results}"
            raise AssertionError(err)


@pytest.mark.parametrize(
    "amino_acid",
    [
        "ALA",
        "GLY",
        "ILE",
        "LEU",
        "PRO",
        "VAL",
        "PHE",
        "TRP",
        "TYR",
        "ASP",
        "GLU",
        "ARG",
        "HIS",
        "LYS",
        "SER",
        "THR",
        "CYS",
        "MET",
        "ASN",
        "GLN",
    ],
)
def test_unfolded_sasa(amino_acid):
    """Test unfolded SASA model."""
    ref_models = ReferenceModels()
    val1 = ref_models.residue_area(amino_acid, model="auton")
    val2 = ref_models.residue_area(amino_acid, model="creamer")
    abs_err = np.absolute(val1 - val2)
    rel_err = 2 * abs_err / (val1 + val2)
    _LOGGER.debug(
        f"{amino_acid}: Auton ({val1:.2f}) and Creamer ({val2:.2f}) differ by "
        f"{abs_err:.2f} (relative error {rel_err:e})."
    )
    # 0.10 accounts for rounding error between the two papers
    if abs_err > 0.101:
        if amino_acid != "PHE":
            err = f"Absolute error ({abs_err}) > 0.10 (round-off tolerance)"
            raise AssertionError(err)
        _LOGGER.error(
            f"{amino_acid}: Auton ({val1:.2f}) & Creamer ({val2:.2f}) differ "
            f"by {abs_err:.2f} (relative error {rel_err:e}) -- error in Auton "
            f"value?"
        )
