"""Calculate solvent-accessible surface areas using Lee-Richards method."""
import numpy as np


def sphere(num):
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
    num = int((np.sqrt(6*num - 32) - 4.0)/6.0)
    line = np.linspace(0, 1, num+2)
    edge = np.delete(line, [0, num+1])
    points = None
    # Generate faces
    for x in [0., 1.]:
        face = np.array([(x, y_, z_) for y_ in edge for z_ in edge])
        if not points is not None:
            points = face
        else:
            points = np.concatenate([points, face])
    for y in [0., 1.]:
        face = np.array([(x_, y, z_) for x_ in edge for z_ in edge])
        points = np.concatenate([points, face])
    for z in [0., 1.]:
        face = np.array([(x_, y_, z) for x_ in edge for y_ in edge])
        points = np.concatenate([points, face])
    # Generate corners
    for x in [0., 1.]:
        for y in [0., 1.]:
            for z in [0, 1.]:
                corner = np.array([x, y, z])
                points = np.vstack((points, corner))
    # Generate edges
    for x in [0., 1.]:
        for y in [0., 1.]:
            edge_ = np.array([(x, y, z_) for z_ in edge])
            points = np.concatenate([points, edge_])
            for z in [0, 1.]:
                edge_ = np.array([(x, y_, z) for y_ in edge])
                points = np.concatenate([points, edge_])
                edge_ = np.array([(x_, y, z) for x_ in edge])
                points = np.concatenate([points, edge_])
    # Move center to origin
    trans = np.array([0.5, 0.5, 0.5])
    points = points - trans
    # Scale to sphere
    dist = np.linalg.norm(points, axis=1)
    points = np.divide(points, dist[:,None])
    return points
