import sys
import gstools as gs
import numpy as np


MODEL_NAME = ["Gaussian", "Exponential", "Matern", "Linear"]

class BaseClass:
    """ Base class for Random Fields """
    def __init__(self, model_name: str, theta: float, anisotropy: list, angles: list, seed: int) -> None:
        """
        Initialise random fields

        Parameters:
        -----------
        model_name: str
            Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
        theta: float
            The scale of the fluctuation
        anisotropy: list
            The anisotropy of the model
        angles: list
            The angles of the model
        seed: int
            The seed number for the random number generator
        """
        # check dimension
        if len(anisotropy) != len(angles) + 1:
            sys.exit(
                'ERROR: angles dimensions need to be dimensions of anisotropy - 1')

        # scale of fluctuation
        self.len_scale = np.array(anisotropy) * theta
        # angles
        self.angles = angles
        # dimensions
        self.n_dim = len(self.len_scale)
        # seed number
        self.seed = seed
        # random fields
        self.random_field = None
        self.random_field_model = None
        # model name
        if model_name not in MODEL_NAME:
            sys.exit(f"ERROR: model_name needs to be: {', '.join(MODEL_NAME)}")
        self.model_name = model_name

    def define_model(self, mean: float, variance: float):
        """
        Define and check model for the Random Field

        Parameters:
        -----------
        mean: float
            The mean of the model
        variance: float
            The variance of the model
        """

        # initialise model
        if self.model_name == 'Gaussian':
            model = gs.Gaussian(dim=self.n_dim, var=variance,
                                len_scale=self.len_scale, angles=self.angles)
        elif self.model_name == 'Exponential':
            model = gs.Exponential(
                dim=self.n_dim, var=variance, len_scale=self.len_scale, angles=self.angles)
        elif self.model_name == 'Matern':
            model = gs.Matern(dim=self.n_dim, var=variance, len_scale=self.len_scale, angles=self.angles)
        elif self.model_name == 'Linear':
            model = gs.Linear(dim=self.n_dim, var=variance, len_scale=self.len_scale, angles=self.angles)
        else:
            sys.exit(f'ERROR: model name: {self.model_name} is not supported')

        self.random_field_model = model


class RandomFields(BaseClass):
    """
    Generate random fields
    """

    def __init__(self, model_name: str, theta: float, anisotropy: list, angles: list, seed: int = 14) -> None:
        """
        Initialise generation of random fields
        """
        super().__init__(model_name, theta, anisotropy, angles, seed)

    def generate(self, nodes: list, mean: float, variance: float) -> None:
        """
        Generate random field

        Parameters:
        ------------
        nodes: np.array
            The nodes of the random field
        mean: float
            The mean of the random field
        variance: float
            The variance of the random field
        """

        # check dimensions of nodes
        if len(nodes) != self.n_dim:
            sys.exit('ERROR: dimensions of nodes do not match dimensions of model')

        # assign model
        self.define_model(mean, variance)

        # create random field
        self.random_field_model = gs.SRF(self.random_field_model, mean=mean, seed=self.seed)
        self.random_field_model(nodes)

        # random field
        self.random_field = self.random_field_model[0]


class ConditionalRandomFields(BaseClass):
    """
    Generate conditional random fields
    """

    def __init__(self, model_name: str, theta: float, anisotropy: list, angles: list, seed: int = 14) -> None:
        """
        Initialise generation of random fields
        """
        super().__init__(model_name, theta, anisotropy, angles, seed)

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
        krige = gs.krige.Ordinary(self.random_field_model, coordinates, data)
        cond_srf = gs.CondSRF(krige)
        cond_srf.set_pos(nodes)

        # random field
        self.random_field = cond_srf(seed=self.seed)
