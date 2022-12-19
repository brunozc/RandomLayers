import sys
import numpy as np
from scipy.spatial import ConvexHull, Delaunay


# tolerance to find distances
TOL = 1e-12

class Mesh:
    def __init__(self) -> None:
        self.number_polygons = []
        self.polygons_points = []


class LayersMesh:
    def __init__(self, planes: list, planes_cov: list, seed: int = 14) -> None:
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

        # generate polygons
        self.generate_polygons()

    def generate_polygons(self) -> None:
        """
        Generate polygons from planes
        """
        # update coordinates of planes with coefficient of variation
        for i, pl in enumerate(self.planes):
            standard_dev = pl * self.planes_cov[i]
            self.plane_points.append(np.random.normal(pl, standard_dev))

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

        for n in nodes:
            aux = False
            # find which polygon the node is in
            for idx, poly in enumerate(self.polygons):
                if poly.find_simplex(n) >= 0:
                    self.polygons_index.append(idx)
                    aux = True
                    break

            if aux is False:
                print(n)
                sys.exit(f'ERROR: Point {n} not in polygon')

        index_polygons = list(set(self.polygons_index))
        self.mesh = Mesh()
        self.mesh.number_polygons = len(index_polygons)

        for i in index_polygons:
            idx = np.where(np.array(self.polygons_index) == i)[0]
            self.mesh.polygons_points.append(nodes[idx])
