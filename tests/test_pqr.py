"""Test PQR parsing."""
import logging
import pytest
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from osmolytes.pqr import parse_pqr_file, count_residues


_LOGGER = logging.getLogger(__name__)
PQR_DATA = [
    ["ATOM", 1, "N", "MET", "1", -22.600, 7.684, -9.315, -0.3200, 2.0000],
    ["ATOM", 2, "CA", "MET", "1", -21.455, 8.620, -9.099, 0.3300, 2.0000],
    ["ATOM", 3, "C", "MET", "1", -21.898, 10.062, -8.899, 0.5500, 1.7000],
    ["ATOM", 4, "O", "MET", "1", -22.362, 10.721, -9.827, -0.5500, 1.4000],
    ["ATOM", 5, "CB", "MET", "1", -20.471, 8.544, -10.268, 0.0000, 2.0000],
    ["ATOM", 6, "CG", "MET", "1", -19.558, 7.330, -10.209, 0.2650, 2.0000],
    ["ATOM", 7, "SD", "MET", "1", -18.624, 7.277, -8.646, -0.5300, 1.8500],
    ["ATOM", 8, "CE", "MET", "1", -16.958, 7.708, -9.204, 0.2650, 2.0000],
    ["ATOM", 9, "HE1", "MET", "1", -16.809, 7.326, -10.114, 0.0000, 0.0000],
    ["ATOM", 10, "HE2", "MET", "1", -16.871, 8.702, -9.236, 0.0000, 0.0000],
    ["ATOM", 11, "HE3", "MET", "1", -16.291, 7.330, -8.565, 0.0000, 0.0000],
    ["ATOM", 12, "H2", "MET", "1", -22.539, 6.944, -8.650, 0.3300, 0.0000],
    ["ATOM", 13, "H3", "MET", "1", -23.453, 8.186, -9.193, 0.3300, 0.0000],
    ["ATOM", 14, "HG2", "MET", "1", -20.120, 6.488, -10.314, 0.0000, 0.0000],
    ["ATOM", 15, "HG3", "MET", "1", -18.920, 7.356, -11.001, 0.0000, 0.0000],
    ["ATOM", 16, "H", "MET", "1", -22.546, 7.319, -10.241, 0.3300, 0.0000],
    ["ATOM", 17, "HA", "MET", "1", -20.974, 8.313, -8.264, 0.0000, 0.0000],
    ["ATOM", 18, "HB3", "MET", "1", -19.900, 9.372, -10.269, 0.0000, 0.0000],
    ["ATOM", 19, "HB2", "MET", "1", -20.989, 8.512, -11.130, 0.0000, 0.0000],
    ["ATOM", 20, "N", "GLN", "2", -21.741, 10.545, -7.674, -0.4000, 1.5000],
    ["ATOM", 21, "CA", "GLN", "2", -22.113, 11.910, -7.330, -0.0000, 2.0000],
    ["ATOM", 22, "C", "GLN", "2", -20.881, 12.616, -6.812, 0.5500, 1.7000],
    ["ATOM", 23, "O", "GLN", "2", -19.835, 11.997, -6.608, -0.5500, 1.4000],
    ["ATOM", 24, "CB", "GLN", "2", -23.203, 11.925, -6.251, 0.0000, 2.0000],
    ["ATOM", 25, "CG", "GLN", "2", -24.481, 11.246, -6.670, 0.0000, 2.0000],
    ["ATOM", 26, "CD", "GLN", "2", -25.494, 11.116, -5.541, 0.5500, 1.7000],
    ["ATOM", 27, "HG2", "GLN", "2", -24.905, 11.773, -7.411, 0.0000, 0.0000],
    ["ATOM", 28, "HA", "GLN", "2", -22.407, 12.370, -8.165, 0.0000, 0.0000],
    ["ATOM", 29, "HG3", "GLN", "2", -24.268, 10.325, -7.007, 0.0000, 0.0000],
    ["ATOM", 30, "H", "GLN", "2", -21.335, 9.873, -6.990, 0.4000, 1.0000],
    ["ATOM", 31, "HB3", "GLN", "2", -23.409, 12.879, -6.023, 0.0000, 0.0000],
    ["ATOM", 32, "HB2", "GLN", "2", -22.848, 11.462, -5.437, 0.0000, 0.0000],
    ["ATOM", 33, "HE22", "GLN", "2", -25.445, 9.103, -5.740, 0.3900, 1.0000],
    ["ATOM", 34, "OE1", "GLN", "2", -25.958, 12.106, -4.969, -0.5500, 1.4000],
    ["ATOM", 35, "NE2", "GLN", "2", -25.846, 9.881, -5.221, -0.7800, 1.5000],
    ["ATOM", 36, "HE21", "GLN", "2", -26.513, 9.707, -4.463, 0.3900, 1.0000],
    ["ATOM", 37, "N", "ARG", "3", -21.000, 13.920, -6.624, -0.4000, 1.5000],
    ["ATOM", 38, "CA", "ARG", "3", -19.885, 14.678, -6.101, -0.0000, 2.0000],
    ["ATOM", 39, "C", "ARG", "3", -20.291, 15.253, -4.763, 0.5500, 1.7000],
    ["ATOM", 40, "O", "ARG", "3", -21.384, 15.807, -4.617, -0.5500, 1.4000],
]
PROTEIN_PATH = Path("tests/data/proteins")


@pytest.mark.parametrize(
    "pqr_path,chain_id",
    [("NikR_no-chains.pqr", None), ("NikR_chains.pqr", "A")],
)
def test_parsing(pqr_path, chain_id):
    """Test PQR parsing with and without chains"""
    pqr_path = PROTEIN_PATH / pqr_path
    with open(pqr_path) as pqr_file:
        atoms = parse_pqr_file(pqr_file)
        for iatom in range(40):
            test_atom = atoms[iatom]
            ref_atom = PQR_DATA[iatom]
            assert test_atom.entry_type == ref_atom[0]
            assert test_atom.pqr_atom_num == ref_atom[1]
            assert test_atom.atom_name == ref_atom[2]
            assert test_atom.res_name == ref_atom[3]
            assert test_atom.res_num == ref_atom[4]
            assert test_atom.chain_id == chain_id
            ref_pos = np.array([ref_atom[5], ref_atom[6], ref_atom[7]])
            np.testing.assert_almost_equal(
                test_atom.position, ref_pos, decimal=3
            )
            np.testing.assert_almost_equal(
                test_atom.charge, ref_atom[8], decimal=3
            )
            np.testing.assert_almost_equal(
                test_atom.radius, ref_atom[9], decimal=3
            )


@pytest.mark.parametrize("protein", ["1A6F", "1STN", "2BU4"])
def test_residue_count(protein):
    """Test aggregate SASAs per residue type as reported in Auton and Bolen
    (doi:10.1073/pnas.0507053102, Supporting Table 2)."""
    pqr_path = PROTEIN_PATH / f"{protein}.pqr"
    with open(pqr_path, "rt") as pqr_file:
        atoms = parse_pqr_file(pqr_file)
    test_counts = count_residues(atoms).sort_index()
    ref_path = PROTEIN_PATH / "Auton-Bolen.yaml"
    with open(ref_path, "rt") as ref_file:
        ref_dict = yaml.load(ref_file, Loader=yaml.FullLoader)
    ref_dict = ref_dict[protein]
    ref_df = pd.DataFrame(ref_dict).T
    ref_df = ref_df[ref_df["number"] > 0]
    ref_counts = np.array(ref_df["number"].sort_index())
    results_df = pd.DataFrame({"Test": test_counts, "Ref": ref_counts})
    _LOGGER.info(f"Results for {protein}:\n{results_df}")
    np.testing.assert_equal(test_counts, ref_counts)


@pytest.mark.parametrize("protein", ["2PTN", "1OVA"])
def test_insertion_codes(protein):
    """Test whether proteins with insertion codes break the software."""
    pqr_path = PROTEIN_PATH / f"{protein}.pqr"
    with open(pqr_path, "rt") as pqr_file:
        parse_pqr_file(pqr_file)
