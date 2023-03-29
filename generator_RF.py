import os
import shutil
import numpy as np

# RandomLayers package
from RandomLayers.random_fields import RandomFields
from RandomLayers.methods import ModelName
from RandomLayers.mesh import LayersMesh
from RandomLayers.utils import slice



def generate_data(x_max, y_max, z_max, nb_max_layers, min_layer_thickness,
                  planes_dict, soils_dict, output_folder, nb_realisations, min_value=1):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    for n in range(nb_realisations):
        try:
            print(f"Number of realisation: {n}")
            np.random.seed(n)

            nb_layers = np.random.randint(1, nb_max_layers + 1)
            idx_to_use_soil = np.random.choice(range(nb_layers), nb_layers, replace=False)
            # print(f"Number of layers: {nb_layers}")

            z_middle = []
            z_low = 0
            for i in range(nb_layers - 1):
                new_z = np.round(np.random.uniform(z_low, z_max), 1)
                if new_z == z_max:
                    break
                z_middle.append(new_z)
                z_low = z_middle[i] + min_layer_thickness
                if z_low >= z_max:
                    break

            nb_layers = int(len(z_middle) + 1)
            idx_to_use_soil = idx_to_use_soil[:nb_layers]
            # print(f"Number of layers: {nb_layers}")

            planes = []
            plane_cov = []
            theta = []
            aniso = []
            angles = []
            planes.append(np.array([[0, 0, 0],
                                [x_max, 0, 0],
                                [x_max, y_max, 0],
                                [0, y_max, 0],
                                ])
            )
            plane_cov.append([0, 0, 0])

            for z in z_middle:
                planes.append(np.array([[0, 0, z],
                                        [x_max, 0, z],
                                        [x_max, y_max, z],
                                        [0, y_max, z],
                                        ])
                            )
                _cov = np.round(np.random.uniform(0, planes_dict["max_cov"]), 3)
                _theta = np.round(np.random.uniform(0, planes_dict["max_theta"]), 3)
                plane_cov.append([0, 0, _cov])
                theta.append([_theta, _theta, _theta])
                aniso.append([1, 1, 1])
                angles.append([0, 0, 0])


            planes.append(np.array([[0, 0, z_max],
                                [x_max, 0, z_max],
                                [x_max, y_max, z_max],
                                [0, y_max, z_max],
                                ])
                        )
            plane_cov.append([0, 0, 0])
            theta.append([1, 1, 1])
            aniso.append([1, 1, 1])
            angles.append([0, 0, 0])

            resample_points_x = 51
            resample_points_y = 6
            resample_points_z = 21

            layers = LayersMesh(planes, plane_cov,
                                model=ModelName.Gaussian, theta=theta, anisotropy=aniso, angles=angles, resample_points=51, seed=n)

            x = np.linspace(0, x_max, resample_points_x)
            y = np.linspace(0, y_max, resample_points_y)
            z = np.linspace(0, z_max, resample_points_z)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

            layers.generate_mesh(np.array([X.ravel(), Y.ravel(), Z.ravel()]).T)

            theta = []
            aniso = []
            angles = []
            for i in range(nb_layers):
                theta.append([np.round(np.random.uniform(min_value, soils_dict["max_theta"][idx_to_use_soil[i]]), 2),
                              np.round(np.random.uniform(min_value, soils_dict["max_theta"][idx_to_use_soil[i]]), 2),
                              np.round(np.random.uniform(min_value, soils_dict["max_theta"][idx_to_use_soil[i]]), 2),
                              ])
                aniso.append([np.round(np.random.uniform(min_value, soils_dict["max_aniso"][idx_to_use_soil[i]]), 2),
                              np.round(np.random.uniform(min_value, soils_dict["max_aniso"][idx_to_use_soil[i]]), 2),
                              np.round(np.random.uniform(min_value, soils_dict["max_aniso"][idx_to_use_soil[i]]), 2)
                              ])
                angles.append([np.round(np.random.uniform(-soils_dict["max_angle"][idx_to_use_soil[i]], soils_dict["max_angle"][idx_to_use_soil[i]]), 2),
                               np.round(np.random.uniform(-soils_dict["max_angle"][idx_to_use_soil[i]], soils_dict["max_angle"][idx_to_use_soil[i]]), 2),
                               np.round(np.random.uniform(-soils_dict["max_angle"][idx_to_use_soil[i]], soils_dict["max_angle"][idx_to_use_soil[i]]), 2),
                               ])

            rf = RandomFields(ModelName.Gaussian, layers.mesh.polygons_points, theta, aniso, angles, seed=n)
            rf.generate(np.array(soils_dict["soil_properties"])[idx_to_use_soil], np.array(soils_dict["soil_var"])[idx_to_use_soil])

            # plot3D(layers.mesh.polygons_points, rf.random_field, output_folder="./", output_name="RF.png")
            # plot3D_viewer(layers.mesh.polygons_points, rf.random_field, output_folder="./", output_name="RF.html")
            coord, sliced_rf = slice(layers.mesh.polygons_points, rf.random_field, 1, 0)

            with open(os.path.join(output_folder, f"./slice_{str(n).zfill(3)}.txt"), "w") as f:
                f.write("x;y;z;IC\n")
                for i in range(len(coord)):
                    f.write(f"{coord[i][0]};{coord[i][1]};{sliced_rf[i]}\n")

            # make plot
            import matplotlib.pylab as plt
            fig, ax = plt.subplots(1,1, figsize=(6, 4))
            ax.set_position([0.1, 0.1, 0.7, 0.8])
            im = ax.imshow(sliced_rf[::-1].reshape((resample_points_x, resample_points_z)),
                           vmin=1., vmax=4., cmap="viridis", extent=[0, x_max, 0, z_max])#, aspect="auto")
            cax = fig.add_axes([0.85, 0.45, 0.02, 0.1])
            cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04)
            cbar.set_label("value")
            plt.savefig(os.path.join(output_folder, f"slice_{str(n).zfill(3)}.png"))
            plt.close()
        except:
            print(f"Error in {n}")
            continue


def split_data(data_path, train_folder, validation_folder, train_size=0.8, shuffle=True):
    """Split data into train and validation set.
    """

    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)

    if not os.path.isdir(validation_folder):
        os.makedirs(validation_folder)

    files = os.listdir(data_path)
    files = [f for f in files if f.endswith(".txt")]

    nb_files = len(files)
    nb_train = int(nb_files * train_size)
    indexes_train = np.random.choice(range(nb_files), nb_train, replace=False)
    indexes_validation = np.array([i for i in range(nb_files) if i not in indexes_train])

    for i in indexes_train:
        shutil.copy(os.path.join(data_path, files[i]), os.path.join(train_folder, files[i]))

    for i in indexes_validation:
        shutil.copy(os.path.join(data_path, files[i]), os.path.join(validation_folder, files[i]))


if __name__ == '__main__':

    planes = {"max_cov": 0.1,
              "max_theta": 10,
              }
    soils = {"max_theta": [10, 5, 10, 5, 10],
             "max_aniso": [10, 5, 10, 5, 10],
             "max_angle": [np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6],
             "soil_properties": [3, 2, 1.5, 4, 2.5],
             "soil_var": [0.3, 0.2, 0.15, 0.4, 0.25],
            }

    x_max = 100
    y_max = 5
    z_max = 10
    nb_max_layer = 5

    min_layer_thickness = 0.1
    output_folder = "./output"
    generate_data(x_max, y_max, z_max, nb_max_layer, min_layer_thickness,
                  planes, soils, output_folder, 1000)

    split_data(output_folder, os.path.join(output_folder, "train"), os.path.join(output_folder, "./validation"),
               train_size=0.8)
