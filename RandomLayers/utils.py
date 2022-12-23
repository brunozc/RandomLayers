import os
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib as mpl


def plot3D(coordinates: list, random_field: list, title: str = "Random Field",
           output_folder = "./", output_name: str = "random_field.png"):

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make plot
    fig = plt.figure(1, figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    ax.set_position([0.1, 0.1, 0.8, 0.8])

    vmin = min(min(aux) for aux in random_field)
    vmax = max(max(aux) for aux in random_field)

    for i, coord in enumerate(coordinates):
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
        ax.scatter(x, y, z, c=random_field[i], vmin=vmin, vmax=vmax, cmap="viridis", edgecolors=None)

    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    ax.set_zlabel('z coordinate')
    norm = mpl.colors.Normalize(vmin=15, vmax=25)
    # Add a colorbar to a plot
    cax = ax.inset_axes([1.1, 0., 0.05, 1])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    fig.suptitle(title)
    plt.savefig(os.path.join(output_folder, output_name))
    plt.close()


def plot2D(coordinates: list, random_field: list, title: str = "Random Field",
           output_folder = "./", output_name: str = "random_field.png"):

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_position([0.1, 0.1, 0.7, 0.8])

    vmin = min(min(aux) for aux in random_field)
    vmax = max(max(aux) for aux in random_field)

    for i, coord in enumerate(coordinates):
        x, y = coord[:, 0], coord[:, 1]
        ax.contourf(x, y, random_field[i].reshape(len(x), len(y)), vmin=vmin, vmax=vmax, cmap="viridis")

    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')

    cax = ax.inset_axes([1.1, 0., 0.05, 1])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    fig.suptitle(title)
    plt.savefig(os.path.join(output_folder, output_name))
    plt.close()


def plot3D_viewer(coordinates: list, random_field: list, title: str = "Random Field",
           output_folder="./", output_name: str = "random_field.html", fps: int = 10, format: str = "html"):

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make plot
    fig = plt.figure(1, figsize=(6, 5))
    ax = fig.add_subplot()
    ax.set_position([0.1, 0.1, 0.8, 0.8])

    vmin = min(min(aux) for aux in random_field)
    vmax = max(max(aux) for aux in random_field)

    # determine unique y
    c_y = np.unique([np.unique(coord[:, 1]) for coord in coordinates])

    plts = []
    for c in c_y:
        img = []
        for i, coord in enumerate(coordinates):
            x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
            idx = np.where(y == c)
            # img.append(ax.imshow(random_field[i][idx].reshape(
                # (len(np.unique(x)), len(np.unique(z)))), vmin=vmin, vmax=vmax, cmap="viridis"))
            img.append(ax.scatter(x[idx], z[idx], c=random_field[i][idx], s=50, marker="s", vmin=vmin, vmax=vmax, cmap="viridis"))
        plts.append(tuple(img))

    ax.set_title(title)
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('z coordinate')
    ax.grid()

    # create animation
    writer = animation.writers[format](fps=fps)
    im_ani = animation.ArtistAnimation(fig, plts,
                                       blit=True)

    # save animation
    im_ani.save(output_name, writer=writer)
