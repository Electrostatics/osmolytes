"""PQR file class.

Parse and store parametrized molecular structure files.
"""
import logging
import numpy as np


_LOGGER = logging.getLogger(__name__)


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
        self.parse_pqr(pqr_string)

    def parse_pqr(self, pqr_string):
        """Parse PQR string for atom.

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
            if words[0] in ["HEADER", "REMARK"]:
                pass
            elif words[0] in ["ATOM", "HETATM"]:
                atoms.append(Atom(line))
            else:
                errstr = f"Unable to parse PQR line: {line}"
                raise ValueError(errstr)
    return atoms
