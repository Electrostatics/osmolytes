"""Test m-value prediction model."""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import linregress
from osmolytes.pqr import parse_pqr_file
from osmolytes.sasa import SolventAccessibleSurface
from osmolytes.energy import transfer_energy


_LOGGER = logging.getLogger(__name__)
PROTEIN_PATH = Path("tests/data/proteins")
# Results (kcal/mol) were extracted visually from the Figure 1 in the paper
# doi: 10.1073/pnas.0507053102
AUTON_RESULTS = {
    "2BU4 urea": -2.40,
    "2BU4 betaine": -0.25,
    "2BU4 proline": -0.05,
    "2BU4 sorbitol": 0.80,
    "2BU4 sucrose": 1.00,
    "2BU4 sarcosine": 1.5,
    "2BU4 TMAO": 1.80,
    "1STN TMAO": 2.85,
    "1A6F TMAO": 3.00,
}


def test_energy(tmp_path):
    """Test the energy model as reported in Auton and Bolen
    (doi:10.1073/pnas.0507053102, Figure 1)"""
    test_results = {}
    for protein in ["1A6F", "1STN", "2BU4"]:
        pqr_path = PROTEIN_PATH / f"{protein}.pqr"
        with open(pqr_path, "rt") as pqr_file:
            atoms = parse_pqr_file(pqr_file)
        xyz_path = Path(tmp_path) / f"{protein}.xyz"
        sas = SolventAccessibleSurface(
            atoms, probe_radius=1.4, num_points=2000, xyz_path=xyz_path
        )
        energy_df = transfer_energy(atoms, sas)
        _LOGGER.info(f"{protein} detailed energies:\n{energy_df.to_string()}")
        energies = energy_df.sum(axis=0).sort_values()
        _LOGGER.info(f"{protein} m-values\n{energies}")
        for osmolyte, value in energies.iteritems():
            key = f"{protein} {osmolyte}"
            if key in AUTON_RESULTS:
                test_results[key] = value
    ref_results = pd.Series(AUTON_RESULTS).sort_index()
    test_results = pd.Series(test_results).sort_index()
    abs_error = np.absolute(ref_results - test_results)
    rel_error = np.divide(abs_error, np.absolute(ref_results))
    err_df = pd.DataFrame(
        {
            "Ref results": ref_results,
            "Test results": test_results,
            "Abs error": abs_error,
            "Rel error": rel_error,
        }
    )
    _LOGGER.info(f"Test results:\n{err_df}")
    results = linregress(ref_results, test_results)
    info = f"Correlation of test with results from paper:  {results}"
    _LOGGER.info(info)
    if not np.isclose(results.pvalue, 0):
        err = f"Poor fit to predicted energies:  {results}"
        raise AssertionError(err)
