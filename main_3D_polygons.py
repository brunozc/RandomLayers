import numpy as np

# RandomLayers package
from RandomLayers.random_fields import RandomFields, ConditionalRandomFields
from RandomLayers.methods import ModelName, KrigingMethod
from RandomLayers.mesh import LayersMesh
from RandomLayers.utils import plot3D, plot3D_viewer, slice


if __name__ == '__main__':

    x_max = 50
    y_max = 20
    z1 = 5
    z2 = 7
    plane_0 = np.array([[0, 0, 0],
                        [x_max, 0, 0],
                        [x_max, y_max, 0],
                        [0, y_max, 0],
                        ])

    plane_1 = np.array([[0, 0, z1],
                        [x_max, 0, z1],
                        [x_max, y_max, z1],
                        [0, y_max, z1],
                        ])

    plane_2 = np.array([[0, 0, z2],
                        [x_max, 0, z2],
                        [x_max, y_max, z2],
                        [0, y_max, z2],
                        ])

    # variance for each plane
    plane_cov = [[0, 0, 0],
                 [0, 0, 0.2],
                 [0, 0, 0],
                ]

    theta = [[1, 1, 1],
             [1, 1, 1]]

    aniso = [[1, 1, 1],
             [1, 1, 1]]

    angles = [[0, 0, 0],
              [0, 0, 0]]

    resample_points_x = 51
    resample_points_z = 41


    layers = LayersMesh([plane_0, plane_1, plane_2], plane_cov,
                        model=ModelName.Gaussian, theta=theta, anisotropy=aniso, angles=angles, resample_points=10)

    x = np.linspace(0, x_max, resample_points_x)
    y = np.linspace(0, y_max, resample_points_x)
    z = np.linspace(0, z2, resample_points_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    layers.generate_mesh(np.array([X.ravel(), Y.ravel(), Z.ravel()]).T)


    theta = [[5, 5, 1],
             [5, 5, 1]]

    aniso = [[1, 1, 1],
             [1, 1, 1]]

    angles = [[-np.pi/6, 0, 0],
              [np.pi/6, 0, 0]]

    var = []

    rf = RandomFields(ModelName.Gaussian, layers.mesh.polygons_points, theta, aniso, angles)
    rf.generate([20, 10], [5, 5])

    plot3D(layers.mesh.polygons_points, rf.random_field, output_folder="./", output_name="RF.png")
    plot3D_viewer(layers.mesh.polygons_points, rf.random_field, output_folder="./", output_name="RF.html")
    coord, sliced_rf = slice(layers.mesh.polygons_points, rf.random_field, 1, 4)

    with open(r"./slice.txt", "w") as f:
        f.write("x;y;z;IC\n")
        for i in range(len(coord)):
            f.write(f"{coord[i][0]};{coord[i][1]};{sliced_rf[i]}\n")

    import matplotlib.pylab as plt
    plt.scatter(np.array(coord)[:, 0], np.array(coord)[:, 1], c=sliced_rf)
    plt.show()