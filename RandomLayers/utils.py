import os
import shutil
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib as mpl


def plot3D(coordinates: list, random_field: list, title: str = "Random Field",
           output_folder = "./", output_name: str = "random_field.png"):
    """
    Plot 3D random field

    Parameters
    ----------
    coordinates : list
        List of coordinates of the random field
    random_field : list
        List of random field values
    title : str
        Title of the plot
    output_folder : str
        Output folder
    output_name : str
        Output fine name
    """
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
    """
    Plot 2D random field

    Parameters
    ----------
    coordinates : list
        List of coordinates of the random field
    random_field : list
        List of random field values
    title : str
        Title of the plot
    output_folder : str
        Output folder
    output_name : str
        Output fine name
    """

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
    """
    Plot a 3D animation of the random field

    Parameters
    ----------
    coordinates : list
        List of coordinates of the random field
    random_field : list
        List of random field values
    title : str
        Title of the plot
    output_folder : str
        Output folder
    output_name : str
        Output fine name
    fps : int
        Frames per second
    format : str
        Format of the animation
    """
    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make plot
    fig = plt.figure(1, figsize=(9, 5))
    ax = fig.add_subplot()
    ax.set_position([0.075, 0.1, 0.8, 0.8])

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
            img.append(ax.scatter(x[idx], z[idx], c=random_field[i][idx], s=75, marker="s", vmin=vmin, vmax=vmax, cmap="viridis"))
        plts.append(tuple(img))

    ax.set_title(title)
    cax = ax.inset_axes([1.02, 0.05, 0.05, .9])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    cbar.set_label("Values")

    ax.set_xlabel('x coordinate')
    ax.set_ylabel('z coordinate')
    ax.grid()

    # create animation
    writer = animation.writers[format](fps=fps)
    im_ani = animation.ArtistAnimation(fig, plts,
                                       blit=True)

    # save animation
    im_ani.save(output_name, writer=writer)
    plt.close()


def slice(coordinates: list, random_field: list, axis: int, coord_slice: float):
    """
    Slice the random field in a given axis at a given coordinate

    Parameters
    ----------
    coordinates : list
        List of coordinates
    random_field : list
        Random field values
    axis : int
        Axis to slice
    coord_slice : float
        Coordinate to slice
    """

    # determine unique coordinates along axis
    c_axis = np.unique([np.unique(coord[:, axis]) for coord in coordinates])
    if coord_slice not in c_axis:
        raise ValueError(f"Coordinate to slice {coord_slice} not in the random field coordinates")

    # determine indexes of remaining coordinates
    indexes = list(range(len(coordinates[0][0])))
    indexes.pop(axis)

    new_coordinates = np.empty((0, len(indexes)))
    slice_data = []
    for i, coord in enumerate(coordinates):

        idx = np.where(coord[:, axis] == coord_slice)
        aux = [coord[idx, j].ravel() for j in indexes]
        new_coordinates = np.append(new_coordinates, np.array(aux).T, axis=0)
        slice_data.extend(random_field[i][idx])

    # Combine coordinates and values into pairs
    combined = list(zip(new_coordinates, slice_data))

    # Sort based on x and y coordinates
    sorted_combined = sorted(combined, key=lambda item: (item[0][0], item[0][1]))

    # Extract sorted coordinates and values
    sorted_coordinates = [item[0] for item in sorted_combined]
    sorted_values = [item[1] for item in sorted_combined]

    return sorted_coordinates, np.array(sorted_values)


def split_data(data_path, train_folder, validation_folder, train_size=0.8, shuffle=True):
    """
    Split data into train and validation set.
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