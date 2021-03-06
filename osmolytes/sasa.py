"""Calculate solvent-accessible surface areas using Lee-Richards method."""
import logging
import pkg_resources
import yaml
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as Tree
from osmolytes.pqr import count_residues


_LOGGER = logging.getLogger(__name__)
# The value at which a radius is considered 0
RADIUS_CUTOFF = 0.0001
# Data files for reference states
CREAMER_DICT = yaml.load(
    pkg_resources.resource_stream(__name__, "data/creamer-areas.yaml"),
    Loader=yaml.FullLoader,
)
AUTON_DICT = yaml.load(
    pkg_resources.resource_stream(__name__, "data/denatured-areas.yaml"),
    Loader=yaml.FullLoader,
)
TRIPEPTIDE_DICT = yaml.load(
    pkg_resources.resource_stream(__name__, "data/tripeptide-areas.yaml"),
    Loader=yaml.FullLoader,
)
# The plastic constant (generalization of golden ratio)
PLASTIC = np.cbrt((9 + np.sqrt(69))/18) + np.cbrt((9 - np.sqrt(69))/18)


def sphere(num):
    """Discretize unit sphere at the origin using a low-discrepancy recurrence.

    Implementation from
    http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    using Kronecker recurrence sequence.

    :param int num:  target number of points on sphere
    :returns:  an array of (num)-by-3 dimension coordinates
    :rtype:  np.ndarray
    """
    n = np.arange(1, num+1, 1)

    a1 = 1.0 / PLASTIC
    a2 = a1 / PLASTIC

    u = (0.5 + a1 * n) % 1
    v = (0.5 + a2 * n) % 1

    theta = np.arccos(2 * u - 1) + 0.5 * np.pi
    phi = 2 * np.pi * v

    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return np.array([x, y, z]).T


class SolventAccessibleSurface:
    """Calculate Lee-Richards solvent-accessible surface area for folded
    proteins."""

    def __init__(
        self, atoms, probe_radius, num_points=1000, xyz_path="surface.xyz"
    ):
        """Initialize the object.

        :param list(Atom) atoms:  list of atoms from which to construct surface
        :param float probe_radius:  radius of probe atom (solvent) in Angstroms
        :param int num_points:  number of points to use for reference sphere
        :param str xyz_path:  path to xyz file (if None, no file is written)
        """
        self.atoms = atoms
        self.probe_radius = probe_radius
        self.num_points = num_points
        self.sphere = sphere(num_points)
        self.max_radius = max([atom.radius for atom in self.atoms])
        self.max_search = 2 * self.max_radius + 2 * probe_radius
        _LOGGER.debug("max_search = %g", self.max_search)
        # Set up atom surface reference spheres
        self.surfaces = []
        for atom in self.atoms:
            if atom.radius < RADIUS_CUTOFF:
                self.surfaces.append(None)
            else:
                atom_sphere = (
                    atom.radius + self.probe_radius
                ) * self.sphere + atom.position
                self.surfaces.append(atom_sphere)
        # Set up tree structure for distance lookup
        self.tree = Tree([atom.position for atom in self.atoms])
        matrix = self.tree.sparse_distance_matrix(
            self.tree, self.max_search, output_type="coo_matrix"
        )
        # Test individual surfaces
        _LOGGER.debug(matrix)
        for i, j, distance in zip(matrix.row, matrix.col, matrix.data):
            if i != j:
                if np.isclose(distance, 0) and np.isclose(
                    self.atoms[i].radius, self.atoms[j].radius
                ):
                    errstr = f"Overlapping atoms ({i}, {j}) of equal radius"
                    _LOGGER.warning(errstr)
                    if i < j:
                        self.surfaces[i] = self.surfaces[i][0::2]
                    else:
                        self.surfaces[i] = self.surfaces[i][1::2]
                else:
                    self.prune_surface(i, j)
        # Dump surface
        if xyz_path is not None:
            self.dump_xyz(xyz_path)

    def dump_xyz(self, xyz_path):
        """Dump surface in XYZ format.

        :param str xyz_path:  path for XYZ-format data
        """
        _LOGGER.debug("Writing debug coordinates to %s", xyz_path)
        fmt = "{name} {x:>8.3f} {y:>8.3f} {z:>8.3f}"
        with open(xyz_path, "wt") as xyz_file:
            for surface in self.surfaces:
                if surface is not None:
                    for point in surface:
                        xyz_file.write(
                            "%s\n"
                            % fmt.format(
                                name="P", x=point[0], y=point[1], z=point[2]
                            )
                        )

    def atom_surface_area(self, iatom):
        """Calculate surface area for this atom.

        :param int iatom:  index of the atom in the atom list
        :returns:  total surface area (Angstroms^2)
        :rtype:  float
        """
        num_ref = np.shape(self.sphere)[0]
        surf = self.surfaces[iatom]
        if surf is not None:
            num_surf = np.shape(surf)[0]
        else:
            num_surf = 0
        atom = self.atoms[iatom]
        tot_radius = atom.radius + self.probe_radius
        area = 4 * np.pi * tot_radius * tot_radius
        return area * float(num_surf) / float(num_ref)

    def surface_area_dictionary(self):
        """Calculate surface area, indexed by chain/residue.

        :returns:  surface area (Angstroms^2) for each residue
        :rtype:  dict
        """
        area_dict = {}
        for iatom, atom in enumerate(self.atoms):
            if atom.chain_id:
                chain_id = f"{atom.chain_id}:"
            else:
                chain_id = ""
            key = f"{chain_id}{atom.res_num}:{atom.res_name}"
            area = self.atom_surface_area(iatom)
            if key in area_dict:
                area_dict[key] = area_dict[key] + area
            else:
                area_dict[key] = area
        return area_dict

    def prune_surface(self, iatom1, iatom2):
        """Prune the surface of atom1 based on the presence of atom2.

        :param int iatom1:  index of first atom
        :param int iatom2:  index of second atom
        """
        atom2 = self.atoms[iatom2]
        if self.surfaces[iatom1] is not None:
            disp12 = self.surfaces[iatom1] - atom2.position
            dist12 = np.sum(disp12 ** 2, axis=1)
            max12 = np.square(atom2.radius + self.probe_radius)
            self.surfaces[iatom1] = self.surfaces[iatom1][dist12 > max12]


class ReferenceModels:
    """Calculate solvent-accessible surface area for reference states.

    Uses data and models from:

    - Creamer TP, Srinivasan R, Rose GD. Modeling unfolded states of proteins
      and peptides. II. Backbone solvent accessibility. Biochemistry. 1997 Mar
      11; 36(10):2832-5. doi: 10.1021/bi962819o. PMID: 9062111.
    - Auton M, Bolen DW. Predicting the energetics of osmolyte-induced protein
      folding/unfolding. Proc Natl Acad Sci U S A. 2005 Oct 18;102(42):
      15065-8. doi: 10.1073/pnas.0507053102. Epub 2005 Oct 7. PMID: 16214887;
      PMCID: PMC1257718.
    """

    def __init__(self):
        self._creamer_dict = CREAMER_DICT
        self._auton_dict = AUTON_DICT
        self._tripeptide_dict = TRIPEPTIDE_DICT

    def residue_area(self, residue, model, how="mean"):
        """Return reference area for the given residue type.

        :param str residue:  3-letter residue name
        :param str model:  model to use

          - creamer: denatured areas from Creamer, Srinivasan, Rose.
            doi: 10.1021/bi962819o
          - auton: denatured areas from Auton, Bolen
            doi: 10.1073/pnas.0507053102
          - tripeptide:  Gly-X-Gly areas for residue X from Auton, Bolen
            doi: 10.1073/pnas.0507053102

        :param str how:  calculation to perform on model values

          - max
          - min
          - mean

        :returns:  model values for sidechain and backbone
        :rtype:  dict
        """
        residue = residue.upper()
        model = model.lower()
        how = how.lower()
        areas = {}
        for what in ["sidechain", "backbone"]:
            if model == "creamer":
                values = self._creamer_dict[residue][what]
            elif model == "auton":
                values = [self._auton_dict[residue][what]]
            elif model == "tripeptide":
                values = [self._tripeptide_dict[residue][what]]
            else:
                err = f"Unknown model: {model}"
                raise ValueError(err)
            if how == "max":
                value = np.max(values)
            elif how == "min":
                value = np.min(values)
            elif how == "mean":
                value = np.mean(values)
            else:
                err = f"Unknown calculation: {how}"
                raise ValueError(err)
            areas[what] = value
        return areas

    def molecule_areas(self, atoms, model, how="mean"):
        """Return reference areas for the molecule.

        :param list(pqr.Atom) residue:  list of atoms in molecule
        :param str model:  model to use

          - creamer: denatured areas from Creamer, Srinivasan, Rose.
            doi: 10.1021/bi962819o
          - auton: denatured areas from Auton, Bolen
            doi: 10.1073/pnas.0507053102
          - tripeptide:  Gly-X-Gly areas for residue X from Auton, Bolen
            doi: 10.1073/pnas.0507053102

        :param str what:  what area to report

          - sidechain
          - backbone
          - total

        :param str how:  calculation to perform on model values

          - max
          - min
          - mean

        :returns:  DataFrame with a row for each residue type
        :rtype:  pd.DataFrame
        """
        count = count_residues(atoms)
        rows = {}
        for res, num in count.iteritems():
            res_areas = self.residue_area(res, model, how)
            row = {}
            for what in ["sidechain", "backbone"]:
                row[what] = num * res_areas[what]
            rows[res] = row
        return pd.DataFrame(rows).T.sort_index()
