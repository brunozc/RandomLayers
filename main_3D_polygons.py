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
    y_max = 20
    z1 = 5
    z2 = 7
    plane_0 = np.array([[0, 0, 0],
                        [x_max, 0, 0],
                        [0, y_max, 0],
                        [x_max, y_max, 0],
                        ])

    plane_1 = np.array([[0, 0, z1],
                        [x_max, 0, z1],
                        [0, y_max, z1],
                        [x_max, y_max, z1],
                        ])

    plane_2 = np.array([[0, 0, z2],
                        [x_max, 0, z2],
                        [0, y_max, z2],
                        [x_max, y_max, z2],
                        ])

    # variance for each plane
    plane_cov = [[0, 0, 0],
                 [0, 0, 0.15],
                 [0, 0, 0],
                ]

    layers = LayersMesh([plane_0, plane_1, plane_2], plane_cov)
    x = np.linspace(0, x_max, 51)
    y = np.linspace(0, y_max, 51)
    z = np.linspace(0, z2, 51)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    layers.generate_mesh(np.array([X.ravel(), Y.ravel(), Z.ravel()]).T)

    theta = [[1, 1, 3],
             [1, 1, 5]]

    anyso = [[1, 1, 1],
             [1, 1, 1]]

    angles = [[np.pi/6, 0, 0],
              [np.pi/6, 0, 0]]

    rf = RandomFields(ModelName.Gaussian, layers.mesh.polygons_points, theta, anyso, angles)
    rf.generate([20, 10], [2, 1])

    plot3D(layers.mesh.polygons_points, rf.random_field, output_folder="./", output_name="RF.png")
