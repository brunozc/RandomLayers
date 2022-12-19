import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# RandomLayers package
from RandomLayers.random_fields import RandomFields, ConditionalRandomFields
from RandomLayers.methods import ModelName, KrigingMethod
from RandomLayers.mesh import LayersMesh
from RandomLayers.utils import plot3D


if __name__ == '__main__':

    x_max = 10
    y1 = 2
    y2 = 5
    z_max = 20
    plane_0 = np.array([[0, 0, 0],
                        [x_max, 0, 0],
                        [0, 0, z_max],
                        [x_max, 0, z_max],
                        ])

    plane_1 = np.array([[0, y1, 0],
                        [x_max, y1, 0],
                        [0, y1, z_max],
                        [x_max, y1, z_max],
                        ])

    plane_2 = np.array([[0, y2, 0],
                        [x_max, y2, 0],
                        [0, y2, z_max],
                        [x_max, y2, z_max],
                        ])

    # variance for each plane
    plane_cov = [[0, 0, 0],
                 [0, .2, 0],
                 [0, 0, 0],
                ]

    layers = LayersMesh([plane_0, plane_1, plane_2], plane_cov)
    x = np.linspace(0, x_max, 11)
    y = np.linspace(0, y2, 11)
    z = np.linspace(0, z_max, 11)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    layers.generate_mesh(np.array([X.ravel(), Y.ravel(), Z.ravel()]).T)

    theta = [[1, 3, 1],
             [1, 5, 1]]

    anyso = [[1, 1, 1],
             [1, 1, 1]]

    angles = [[np.pi/6, 0, 0],
              [np.pi/6, 0, 0]]

    rf = RandomFields(ModelName.Gaussian, layers.mesh.polygons_points, theta, anyso, angles)
    rf.generate([20, 10], [2, 1])

    plot3D(layers.mesh.polygons_points, rf.random_field, output_folder="./", output_name="RF.png")