"""Calculate solvent-accessible surface areas using Lee-Richards method."""
import logging
import numpy as np
from scipy.spatial import cKDTree as Tree
from osmolytes.pqr import Atom


_LOGGER = logging.getLogger(__name__)
# The value at which a radius is considered 0
RADIUS_CUTOFF = 0.0001


def sphere_cube(num):
    """Discretize a sphere using points uniformly distributed on a cube.

    Discretize a unit sphere at the origin with a cube geodesic.

    :param num:  target number of points on sphere
    :type num:  int
    :returns:  an array of (anum)-by-3 dimension where anum may not be
        exactly the same as num
    :rtype:  np.ndarray
    """
    if num < 8:
        num = 8
    num = int((np.sqrt(6 * num - 32) - 4.0) / 6.0)
    line = np.linspace(0, 1, num + 2)
    edge = np.delete(line, [0, num + 1])
    points = None
    # Generate faces
    for x in [0.0, 1.0]:
        face = np.array([(x, y_, z_) for y_ in edge for z_ in edge])
        if not points is not None:
            points = face
        else:
            points = np.concatenate([points, face])
    for y in [0.0, 1.0]:
        face = np.array([(x_, y, z_) for x_ in edge for z_ in edge])
        points = np.concatenate([points, face])
    for z in [0.0, 1.0]:
        face = np.array([(x_, y_, z) for x_ in edge for y_ in edge])
        points = np.concatenate([points, face])
    # Generate corners
    for x in [0.0, 1.0]:
        for y in [0.0, 1.0]:
            for z in [0, 1.0]:
                corner = np.array([x, y, z])
                points = np.vstack((points, corner))
    # Generate edges
    for x in [0.0, 1.0]:
        for y in [0.0, 1.0]:
            edge_ = np.array([(x, y, z_) for z_ in edge])
            points = np.concatenate([points, edge_])
            for z in [0, 1.0]:
                edge_ = np.array([(x, y_, z) for y_ in edge])
                points = np.concatenate([points, edge_])
                edge_ = np.array([(x_, y, z) for x_ in edge])
                points = np.concatenate([points, edge_])
    # Move center to origin
    trans = np.array([0.5, 0.5, 0.5])
    points = points - trans
    # Scale to sphere
    dist = np.linalg.norm(points, axis=1)
    points = np.divide(points, dist[:, None])
    return points


def sphere_cylinder(num):
    """Discretize a sphere using cylinder transform.

    Discretize a unit sphere at the origin.

    .. todo:: Fix duplicate points at poles

    :param num:  target number of points on sphere
    :type num:  int
    :returns:  an array of (anum)-by-3 dimension where anum may not be
        exactly the same as num
    :rtype:  np.ndarray
    """
    num = int(np.sqrt(num))
    theta = np.linspace(0, 2 * np.pi, num, endpoint=False)
    u = np.linspace(-1, 1, num)
    theta_z = np.array([(t, z) for t in theta for z in u])
    cos_theta = np.cos(theta_z[:, 0])
    sin_theta = np.sin(theta_z[:, 0])
    z2 = np.square(theta_z[:, 1])
    r = np.sqrt(1 - z2)
    x = r * cos_theta
    y = r * sin_theta
    z = theta_z[:, 1]
    return np.array([x, y, z]).T


class SolventAccessibleSurface:
    """Class for a Lee-Richards solvent-accessible surface."""

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
        self.sphere = sphere_cube(num_points)
        self.max_radius = max([atom.radius for atom in self.atoms])
        self.max_search = self.max_radius + 2 * probe_radius
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
        for i, j, distance in zip(matrix.row, matrix.col, matrix.data):
            if i != j:
                self.prune_surfaces(i, j)
        # Dump surface
        if xyz_path is None:
            self.dump_xyz(xyz_path)

    def dump_xyz(self, xyz_path):
        """Dump surface in XYZ format.

        :param str xyz_path:  path for XYZ-format data
        """
        _LOGGER.warning("Writing debug coordinates to %s", xyz_path)
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

    def prune_surfaces(self, iatom1, iatom2):
        """Prune the surfaces for the specified atoms based on overlap.

        :param int iatom1:  index of first atom
        :param int iatom2:  index of second atom
        """
        atom1 = self.atoms[iatom1]
        atom2 = self.atoms[iatom2]
        if self.surfaces[iatom1] is not None:
            # new_surf = []
            # for surf in self.surfaces[iatom1]:
            #     disp = surf - atom2.position
            #     dist2 = np.sum(disp ** 2)
            #     max2 = np.square(atom2.radius + self.probe_radius)
            #     if dist2 > max2:
            #         new_surf.append(surf)
            # self.surfaces[iatom1] = new_surf
            disp12 = self.surfaces[iatom1] - atom2.position
            dist12 = np.sum(disp12 ** 2, axis=1)
            max12 = np.square(atom2.radius + self.probe_radius)
            self.surfaces[iatom1] = self.surfaces[iatom1][dist12 > max12]
        if self.surfaces[iatom2] is not None:
            # new_surf = []
            # for surf in self.surfaces[iatom2]:
            #     disp = surf - atom1.position
            #     dist2 = np.sum(disp ** 2)
            #     max2 = np.square(atom1.radius + self.probe_radius)
            #     if dist2 > max2:
            #         new_surf.append(surf)
            # self.surfaces[iatom2] = new_surf
            disp21 = self.surfaces[iatom2] - atom1.position
            dist21 = np.sum(disp21 ** 2, axis=1)
            max21 = np.square(atom1.radius + self.probe_radius)
            self.surfaces[iatom2] = self.surfaces[iatom2][dist21 > max21]
