import sys
import gstools as gs
import numpy as np


class BaseClass:
    """ Base class for Random Fields """
    def __init__(self, model_name: str, polygons, theta: list, anisotropy: list, angles: list, seed: int) -> None:
        """
        Initialise random fields

        Parameters:
        -----------
        model_name: str
            Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
        polygons: list
            The polygons of the mesh
        theta: list
            The scale of the fluctuation
        anisotropy: list
            The anisotropy of the model
        angles: list
            The angles of the model
        seed: int
            The seed number for the random number generator
        """
        # check dimension
        if len(anisotropy) != len(angles):
            sys.exit('ERROR: angles dimensions need to be dimensions of anisotropy.')

        # scale of fluctuation
        self.len_scale = np.array(anisotropy) * theta
        # angles
        self.angles = angles
        # seed number
        self.seed = seed
        # random fields
        self.random_field = []
        self.random_field_model = []
        # model name
        self.model_name = model_name
        # number of polygons
        self.nb_polygons = len(polygons)
        # polygons
        self.polygons = polygons

        # check dimensions
        aux = [i.shape[1] for i in self.polygons]
        for i in aux:
            if i != self.polygons[0].shape[1]:
                sys.exit('ERROR: all polygons need to have the same dimension.')
        self.n_dim = aux[0]


    def define_model(self, mean: list, variance: list):
        """
        Define and check model for the Random Field
        Creates a model for each polygon

        Parameters:
        -----------
        mean: float
            The mean of the model
        variance: float
            The variance of the model
        """

        # for each polygon generate RF model
        for i in range(self.nb_polygons):
            # initialise model
            if self.model_name.value == 'Gaussian':
                model = gs.Gaussian(dim=self.n_dim, var=variance[i], len_scale=self.len_scale[i], angles=self.angles[i])
            elif self.model_name.value == 'Exponential':
                model = gs.Exponential(
                    dim=self.n_dim, var=variance[i], len_scale=self.len_scale[i], angles=self.angles[i])
            elif self.model_name.value == 'Matern':
                model = gs.Matern(dim=self.n_dim, var=variance[i], len_scale=self.len_scale[i], angles=self.angles[i])
            elif self.model_name.value == 'Linear':
                model = gs.Linear(dim=self.n_dim, var=variance[i], len_scale=self.len_scale[i], angles=self.angles[i])
            else:
                sys.exit(f'ERROR: model name: {self.model_name.value} is not supported')

            self.random_field_model.append(model)


class RandomFields(BaseClass):
    """
    Generate random fields
    """

    def __init__(self, model_name: str, polygons: list, theta: list, anisotropy: list, angles: list, seed: int = 14) -> None:
        """
        Initialise generation of random fields

        Parameters:
        -----------
        model_name: str
            Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
        polygons: list
            The polygons for the random field
        theta: list
            The scale of the fluctuation
        anisotropy: list
            The anisotropy of the model
        angles: list
            The angles of the model
        seed: int
            The seed number for the random number generator
        """
        super().__init__(model_name, polygons, theta, anisotropy, angles, seed)

    def generate(self, mean: list, variance: list) -> None:
        """
        Generate random field

        Parameters:
        ------------
        mean: list
            The mean of the random field for each polygon
        variance: list
            The variance of the random field for each polygon
        """

        # check dimensions of nodes
        for nodes in self.polygons:
            if nodes.shape[1] != self.n_dim:
                sys.exit('ERROR: dimensions of nodes do not match dimensions of model')

        # assign model
        self.define_model(mean, variance)

        # create random field
        for i, nodes in enumerate(self.polygons):
            self.random_field_model[i] = gs.SRF(self.random_field_model[i], mean=mean[i], seed=self.seed)
            self.random_field_model[i](nodes.T)
            # random field
            self.random_field.append(self.random_field_model[i][0])


class ConditionalRandomFields(BaseClass):
    """
    Generate conditional random fields
    """

    def __init__(self, model_name: str, kriging_model: str, theta: float, anisotropy: list, angles: list, seed: int = 14) -> None:
        """
        Initialise generation of random fields

        Parameters:
        -----------
        model_name: str
            Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
        kriging_model: str
            Name of the kriging model to be used. Options are: "Ordinary", "Universal", "Simple"
        theta: float
            The scale of the fluctuation
        anisotropy: list
            The anisotropy of the model
        angles: list
            The angles of the model
        seed: int
            The seed number for the random number generator
        """
        super().__init__(model_name, theta, anisotropy, angles, seed)
        self.kriging = kriging_model


    def generate(self, nodes: list, mean: float, variance: float, coordinates: list, data: list) -> None:
        """
        Generate conditional random field

        Parameters:
        ------------
        nodes: np.array
            The nodes of the random field
        mean: float
            The mean of the random field
        variance: float
            The variance of the random field
        coordinates: list
            The coordinates of the data
        data: list
            The data to condition the random field
        """

        # check dimensions of nodes
        if len(nodes) != self.n_dim:
            sys.exit('ERROR: dimensions of nodes do not match dimensions of model')

        # assign model
        self.define_model(mean, variance)

        # create conditional random field
        if self.kriging.value == 'Ordinary':
            krige = gs.krige.Ordinary(self.random_field_model, coordinates, data)
        elif self.kriging.value == 'Simple':
            krige = gs.krige.Simple(self.random_field_model, coordinates, data)
        else:
            sys.exit(f'ERROR: kriging model name: {self.kriging.value} is not supported')

        cond_srf = gs.CondSRF(krige)
        cond_srf.set_pos(nodes)

        # random field
        self.random_field = cond_srf(seed=self.seed)
