import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl

# RandomLayers package
from RandomLayers.random_fields import RandomFields, ConditionalRandomFields
from RandomLayers.methods import ModelName, KrigingMethod
from RandomLayers.utils import plot2D

if __name__ == '__main__':
    # regular grid
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    X, Y = np.meshgrid(x, y, indexing="ij")

    rf = RandomFields(ModelName.Gaussian, [np.array([X.ravel(), Y.ravel()]).T],
                      theta=[[1, 3]], anisotropy=[[1, 1]], angles=[[np.pi/6, 0]])
    rf.generate([20], [2])

    # non regular grid
    X, Y = np.meshgrid(x, y, indexing="ij") + np.random.random((11, 11)) * 0.1
    rf2 = RandomFields(ModelName.Gaussian, [np.array([X.ravel(), Y.ravel()]).T],
                       theta=[[1, 3]], anisotropy=[[1, 1]], angles=[[np.pi/6, 0]])
    rf2.generate([20], [2])

    plot2D([np.array([x, y]).T], rf.random_field, output_folder="./", output_name="RF2.png")


    # # make plot
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].set_position([0.1, 0.1, 0.35, 0.8])
    # ax[1].set_position([0.5, 0.1, 0.35, 0.8])

    # cp = ax[0].contourf(x, y, rf.random_field.reshape(11, 11), vmin=15, vmax=25, cmap="viridis")
    # ax[0].set_xlabel('x coordinate')
    # ax[0].set_ylabel('y coordinate')
    # cp = ax[1].contourf(x, y, rf2.random_field.reshape(11, 11), vmin=15, vmax=25, cmap="viridis")
    # cax = ax[1].inset_axes([1.1, 0., 0.05, 1])
    # norm = mpl.colors.Normalize(vmin=15, vmax=25)
    # # Add a colorbar to a plot
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax[1], cax=cax)
    # ax[1].set_xlabel('x coordinate')
    # fig.suptitle('Random fields')
    # plt.show()

    # # define conditional data
    # x_cond = [2, 6]

    # coordinates = np.empty([22, 2])
    # for i, xi in enumerate(x_cond):
    #     Xi, Yi = np.meshgrid(xi, y, indexing="ij")
    #     coordinates[i * 11: (i + 1) * 11, 0] = Xi
    #     coordinates[i * 11: (i + 1) * 11, 1] = Yi

    # # collect basic rf at this points
    # data = np.vstack([rf.random_field.reshape(11, 11)[3, :],
    #                   rf.random_field.reshape(11, 11)[7, :]])

    # crf = ConditionalRandomFields(ModelName.Gaussian, KrigingMethod.Ordinary,
    #                               [1, 3], [1, 1], [np.pi/6])
    # crf.generate([X, Y], 20, 2, coordinates, data)

    # # make plot
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].set_position([0.1, 0.1, 0.35, 0.8])
    # ax[1].set_position([0.5, 0.1, 0.35, 0.8])
    # cp = ax[0].contourf(x, y, rf.random_field.reshape(11, 11), vmin=15, vmax=25, cmap="viridis")
    # ax[0].set_xlabel('x coordinate')
    # ax[0].set_ylabel('y coordinate')
    # ax[0].set_title("Normal random field")
    # for i, var in enumerate(x_cond):
    #     ax[0].scatter([var] * len(y), y, c=data[i, :], cmap="viridis", vmin=15, vmax=25, edgecolors='k')

    # cp = ax[1].contourf(x, y, crf.random_field.reshape(11, 11), vmin=15, vmax=25, cmap="viridis")
    # cax = ax[1].inset_axes([1.1, 0., 0.05, 1])
    # norm = mpl.colors.Normalize(vmin=15, vmax=25)
    # # Add a colorbar to a plot
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax[1], cax=cax)
    # ax[1].set_xlabel('x coordinate')
    # ax[1].set_title("Conditional random field")
    # for i, var in enumerate(x_cond):
    #     ax[1].scatter([var] * len(y), y, c=data[i, :], cmap="viridis", vmin=15, vmax=25, edgecolors='k')
    # plt.show()
