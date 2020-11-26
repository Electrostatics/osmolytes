"""Test m-value prediction model."""
import logging
from pathlib import Path
import pytest
from osmolytes.pqr import parse_pqr_file
from osmolytes.sasa import SolventAccessibleSurface, ReferenceModels
from osmolytes.energy import transfer_energy


_LOGGER = logging.getLogger(__name__)
PROTEIN_PATH = Path("tests/data/proteins")


@pytest.mark.parametrize("protein", ["1A6F", "1STN", "2BU4"])
def test_energy(protein, tmp_path):
    """Test the energy model."""
    pqr_path = PROTEIN_PATH / f"{protein}.pqr"
    with open(pqr_path, "rt") as pqr_file:
        atoms = parse_pqr_file(pqr_file)
    xyz_path = Path(tmp_path) / f"{protein}.xyz"
    sas = SolventAccessibleSurface(
        atoms, probe_radius=1.4, num_points=2000, xyz_path=xyz_path
    )
    energy_df = transfer_energy(atoms, sas)
    raise NotImplementedError()