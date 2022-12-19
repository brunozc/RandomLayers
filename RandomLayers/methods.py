from enum import Enum


class ModelName(Enum):
    Gaussian = "Gaussian"
    Exponential = "Exponential"
    Matern = "Matern"
    Linear = "Linear"


class Krigging(Enum):
    Ordinary = "Ordinary"
    Universal = "Universal"
    Simple = "Simple"
