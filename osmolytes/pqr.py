"""PQR file class.

Parse and store parametrized molecular structure files.
"""
import logging
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd


_LOGGER = logging.getLogger(__name__)


BACKBONE_ATOMS = {"OXT", "H2", "H3", "H", "C", "O", "N", "HA", "HA2", "CA"}
SIDECHAIN_ATOMS = {
    "CB",
    "HH21",
    "OE1",
    "HZ",
    "HD22",
    "HZ1",
    "HH22",
    "CZ",
    "OE2",
    "NE",
    "CE1",
    "CD",
    "HG2",
    "HB1",
    "HD12",
    "NE2",
    "HG12",
    "SG",
    "HH11",
    "HB",
    "HE21",
    "HD23",
    "CG1",
    "HA3",
    "HE22",
    "HD11",
    "OH",
    "HG23",
    "HE",
    "HH",
    "HD21",
    "SD",
    "OG",
    "HZ3",
    "HB2",
    "HD3",
    "NH1",
    "HH12",
    "HD2",
    "HG1",
    "NZ",
    "HE2",
    "OD1",
    "HB3",
    "ND1",
    "CG2",
    "HG3",
    "HE3",
    "HG13",
    "CE",
    "OD2",
    "HE1",
    "HG21",
    "NH2",
    "HD13",
    "CD2",
    "HG22",
    "OG1",
    "CD1",
    "HG",
    "CE2",
    "ND2",
    "HZ2",
    "HD1",
    "CG",
    "HG11",
    "CZ2",
    "CE3",
    "CZ3",
    "NE1",
    "HH2",
    "CH2",
}


class Atom:
    """Atom class for storing PQR information."""

    def __init__(self, pqr_string=None):
        """Initialize Atom class.

        :param str pqr_string:  line from PQR file for initializing structure
        """
        self.pqr_string = pqr_string
        self.entry_type = None
        self.pqr_atom_num = None
        self.atom_name = None
        self.res_name = None
        self.chain_id = None
        self.res_num = None
        self.ins_code = None
        self.position = None
        self.charge = None
        self.radius = None
        self.neighbors = []
        if pqr_string is not None:
            self.parse_pqr(pqr_string)

    def parse_pqr(self, pqr_string):
        """Parse PQR string for atom.

        .. todo:: look for charge variations not captured by naming scheme
                  (e.g., ASH vs. ASP)

        :param str pqr_string:  line from PQR file for initializing structure
        """
        line = pqr_string.strip()
        words = line.split()
        if len(words) > 11:
            errstr = "Too many entries ({num}) in PQR file line: {line}"
            raise ValueError(errstr.format(num=len(words), line=line))
        elif len(words) < 10:
            errstr = "Too few entries ({num}) in PQR file line: {line}"
            raise ValueError(errstr.format(num=len(words), line=line))
        elif len(words) == 11:
            has_chain = True
        else:
            has_chain = False
        self.entry_type = words[0]
        self.pqr_atom_num = int(words[1])
        self.atom_name = words[2]
        self.res_name = words[3]
        if has_chain:
            self.chain_id = words[4]
            chain_offset = 1
        else:
            chain_offset = 0
        self.res_num = int(words[4 + chain_offset])
        x = float(words[5 + chain_offset])
        y = float(words[6 + chain_offset])
        z = float(words[7 + chain_offset])
        self.position = np.array([x, y, z])
        self.charge = float(words[8 + chain_offset])
        self.radius = float(words[9 + chain_offset])

    def __str__(self):
        outstr = ""
        tstr = self.entry_type
        outstr += str.ljust(tstr, 6)[:6]
        tstr = f"{self.pqr_atom_num:d}"
        outstr += str.rjust(tstr, 5)[:5]
        outstr += " "
        tstr = self.atom_name
        if len(tstr) == 4 or len(tstr.strip("FLIP")) == 4:
            outstr += str.ljust(tstr, 4)[:4]
        else:
            outstr += " " + str.ljust(tstr, 3)[:3]
        tstr = self.res_name
        if len(tstr) == 4:
            outstr += str.ljust(tstr, 4)[:4]
        else:
            outstr += " " + str.ljust(tstr, 3)[:3]
        outstr += " "
        tstr = self.chain_id if self.chain_id else ""
        outstr += str.ljust(tstr, 1)[:1]
        tstr = f"{self.res_num:d}"
        outstr += str.rjust(tstr, 4)[:4]
        outstr += f"{self.ins_code}   " if self.ins_code else "    "
        tstr = f"{self.position[0]:8.3f}"
        outstr += str.ljust(tstr, 8)[:8]
        tstr = f"{self.position[1]:8.3f}"
        outstr += str.ljust(tstr, 8)[:8]
        tstr = f"{self.position[2]:8.3f}"
        outstr += str.ljust(tstr, 8)[:8]
        ffcharge = (
            f"{self.charge:.4f}" if self.charge is not None else "0.0000"
        )
        outstr += str.rjust(ffcharge, 8)[:8]
        ffradius = (
            f"{self.radius:.4f}" if self.radius is not None else "0.0000"
        )
        outstr += str.rjust(ffradius, 7)[:7]
        return outstr

    def distance2(self, other):
        """Return the squared distance between this atom and another

        :param Atom other:  other atom for computing distance
        :returns:  squared distance
        :rtype:  float
        """
        displacement = self.position - other.position
        return np.inner(displacement, displacement)


def count_residues(atoms):
    """Count residues by type

    :param list(Atom) atoms:  array of atoms to count
    :returns:  Series with counts
    :rtype:  pd.Series
    """
    res_set = {(atom.res_num, atom.res_name) for atom in atoms}
    counts = Counter()
    for res_tup in res_set:
        res_type = res_tup[1]
        counts.update([res_type])
    return pd.Series(counts).sort_index()


def aggregate(
    atoms, data, chain_id=True, res_name=True, res_num=True, sidechain=True
):
    """Aggregate the array of data into a dictionary.

    .. todo::  This is a horrible kludge.

    :param list(Atom) atoms:  array of atoms over which to aggregate data
    :param list(float) data:  array of data to aggregate
    :param bool chain_id:  whether to aggregate by chain
    :param bool res_name:  whether to aggregate by residue type
    :param bool res_num:  whether to aggregate by residue number
    :param bool sidechain:  whether to aggregate by sidechain/backbone
    :returns:  DataFrame with aggregated data
    :rtype:  pd.DataFrame
    """
    aggregate = OrderedDict()
    unknown_atoms = set()
    unknown_error = False
    for iatom, atom in enumerate(atoms):
        key = []
        if chain_id:
            key.append(atom.chain_id)
        if res_name:
            key.append(atom.res_name)
        if res_num:
            key.append(str(atom.res_num))
        if sidechain:
            if atom.atom_name in BACKBONE_ATOMS:
                key.append("backbone")
            elif atom.atom_name in SIDECHAIN_ATOMS:
                key.append("sidechain")
        key = " ".join(key)
        if key in aggregate:
            aggregate[key] = aggregate[key] + data[iatom]
        else:
            aggregate[key] = data[iatom]
    if unknown_error:
        err = f"Unknown atom types:  {unknown_atoms}"
        raise ValueError(err)
    rows = []
    for key, value in aggregate.items():
        row = key.split()
        row.append(value)
        rows.append(row)
    columns = []
    if chain_id:
        columns.append("chain_id")
    if res_name:
        columns.append("res_name")
    if res_num:
        columns.append("res_num")
    if sidechain:
        columns.append("sidechain")
    columns.append("value")
    df = pd.DataFrame(rows, columns=columns)
    return df


def parse_pqr_file(pqr_file):
    """Parse a PQR file.

    :param file pqr_file:  input file-like object ready for reading
    :returns:  list of atom objects
    :rtype:  list(Atom)
    """
    atoms = []
    for line in pqr_file:
        line = line.strip()
        if line:
            words = line.split()
            if words[0] in [
                "HEADER",
                "REMARK",
                "TER",
                "COMPND",
                "AUTHOR",
                "END",
            ]:
                pass
            elif words[0] in ["ATOM", "HETATM"]:
                atoms.append(Atom(line))
            else:
                errstr = f"Unable to parse PQR line: {line}"
                raise ValueError(errstr)
    return atoms
