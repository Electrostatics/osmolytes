"""Test m-value prediction model."""
import logging
from pathlib import Path
import pytest
from osmolytes.main import main


_LOGGER = logging.getLogger(__name__)
PROTEIN_PATH = Path("tests/data/proteins")


@pytest.mark.parametrize(
    "protein,output_fmt", [("1A6F", "csv"), ("2BU4", "xlsx")]
)
def test_output(protein, output_fmt, tmp_path):
    """Test the output function"""
    pqr_path = PROTEIN_PATH / f"{protein}.pqr"
    xyz_path = Path(tmp_path) / f"{protein}.xyz"
    output_dir = Path(tmp_path)
    main(
        [
            "--solvent-radius",
            "1.4",
            "--surface-points",
            "200",
            "--surface-output",
            str(xyz_path),
            "--output",
            output_fmt,
            "--output-dir",
            str(output_dir),
            str(pqr_path),
        ]
    )
