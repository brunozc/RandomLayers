import numpy as np
from scipy.interpolate import interp2d
from scipy.spatial import Delaunay

# RandomLayers packages
from RandomLayers.random_fields import RandomFields
import RandomLayers.vectors_operation as vectors_operation



class Mesh:
    def __init__(self) -> None:
        self.number_polygons = []
        self.polygons_points = []


class LayersMesh:
    """
    Class to generate a mesh of layers

    The points of the mesh need to be generated counter-clockwise
    """
    def __init__(self, planes: list, planes_cov: list, seed: int = 14,
                model: str = None, theta: list = None, anisotropy: list = None, angles: list = None,
                resample_points: int = 10) -> None:
        """
        Initialise layers

        Parameters:
        -----------
        planes: list
            The planes of the layers
        planes_cov: list
            The coefficient of variation of the planes
        seed: int
            The seed number for the random number generator
        resample_points: int
            The number of points to resample the planes
        """

        self.polygons_points = []
        self.plane_points = []
        self.polygons = []
        self.planes = planes
        self.planes_cov = planes_cov
        self.seed = seed
        np.random.seed(self.seed) # fix seed
        self.polygons_index = []  # index of the polygon the node is in
        self.mesh = None
        self.resample_points = resample_points

        # check if coordinates are clockwise
        for pl in planes:
            coords = vectors_operation.project_point_to_plane(pl, pl)
            if vectors_operation.check_orientation(coords) != "anti-clockwise":
                raise ValueError(f'Points {pl} are not in counter-clockwise order')

        # generate polygons
        self.generate_polygons(model, theta, anisotropy, angles, nb_steps = self.resample_points)

    def generate_polygons(self, model: str, theta: list, anisotropy: list, angles: list, nb_steps: int) -> None:
        """
        Generate polygons from planes
        """

        # update coordinates of planes with coefficient of variation
        for i, pl in enumerate(self.planes):

            # define interpolation of z
            f = interp2d(pl[:, 0], pl[:, 1], pl[:, 2])

            # define lines
            new_points = np.vstack([pl, pl[0]])

            # resample points in plane (knowing it is counter-clockwise)
            resampled_points = []
            for j, _ in enumerate(pl):
                resampled_points.extend(np.linspace(new_points[j], new_points[j + 1], nb_steps))

            # remove repeated points
            resampled_points = np.unique(resampled_points, axis=0)

            # create new coordinates
            xy = np.meshgrid(resampled_points[:, 0], resampled_points[:, 1])
            new_z = f(resampled_points[:, 0], resampled_points[:, 1])
            xyz = np.vstack([xy[0].ravel(), xy[1].ravel(), new_z.ravel()]).T

            # calculate distance from first point
            distance = np.sqrt(np.sum((xyz-xyz[0, :])**2, axis=1))

            # generate random field for each plane
            aux_coord = np.zeros((len(xyz), len(xyz[0])))

            for j, x in enumerate(xyz.T):
                if self.planes_cov[i][j] != 0:
                    rf = RandomFields(model, [distance.reshape(-1, 1)],
                                      [theta[i][j]], [anisotropy[i][j]], [angles[i][j]], seed=self.seed)
                    rf.generate([0], [self.planes_cov[i][j]])
                    aux_coord[:, j] = rf.random_field[0] + x
                else:
                    aux_coord[:, j] = x.ravel()

            self.plane_points.append(aux_coord)

        # collect points for each polygon
        for idx in range(len(self.planes) - 1):
            self.polygons_points.append(np.vstack([self.plane_points[idx],
                                                   self.plane_points[idx + 1]]))

        # make polygons
        for pl in self.polygons_points:
            self.polygons.append(Delaunay(pl))


    def generate_mesh(self, nodes: np.ndarray) -> None:
        """
        Generate index with the polygon where the node is

        Parameters:
        -----------
        nodes: np.array
            The nodes of the mesh
        """

        # find which polygon the node is in
        for n in nodes:
            aux = False
            # find which polygon the node is in
            for idx, poly in enumerate(self.polygons):
                if poly.find_simplex(n, tol=1e-6) >= 0:
                    self.polygons_index.append(idx)
                    aux = True
                    break

            if aux is False:
                raise ValueError(f'Point {n} not in polygon')

        index_polygons = list(set(self.polygons_index))
        self.mesh = Mesh()
        self.mesh.number_polygons = len(index_polygons)

        for i in index_polygons:
            idx = np.where(np.array(self.polygons_index) == i)[0]
            self.mesh.polygons_points.append(nodes[idx])
