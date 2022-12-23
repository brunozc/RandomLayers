import numpy as np


def project_point_to_plane(plane_points: np.array, point_target: np.array):
    """
    Project a point onto a plane

    Parameters:
    -----------
    plane_points: np.array
        Points that form the plane
    point: np.array
        The point to be projected

    Returns:
    --------
    projected_point: np.array
        The projected point
    """

    # define two vectors in the plane
    vector_1 = plane_points[1] - plane_points[0]
    vector_2 = plane_points[2] - plane_points[0]

    # origin of the system
    origin = plane_points[0]

    # define normal vector
    normal_vector = np.cross(vector_1, vector_2)
    normalised_normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # define orthogonal vectors to the normal vector
    x = vector_1

    x = x - x.dot(normal_vector) * normal_vector
    x = x / np.linalg.norm(x)
    y = np.cross(normal_vector, x)
    y = y / np.linalg.norm(y)

    proj_coord = []
    for target in point_target:
        s = np.dot(normalised_normal_vector, target - origin)
        proj_coord.append([np.dot(x, target - origin), np.dot(y, target - origin)])

    return np.array(proj_coord)


def check_orientation(points):
    """
    Check if points are in clockwise or counter-clockwise order

    Parameters:
    -----------
    points: list
        The points to check
    """

    # calculate area of polygon
    area = 0
    for i in range(len(points) - 1):
        area += (points[i][0] * points[i+1][1]) - (points[i+1][0] * points[i][1])
    area += (points[-1][0] * points[0][1]) - (points[0][0] * points[-1][1])
    area = area / 2

    if area > 0:
        return "anti-clockwise"
    else:
        return "clockwise"
