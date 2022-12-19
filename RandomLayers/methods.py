from enum import Enum


class ModelName(Enum):
    Gaussian = "Gaussian"
    Exponential = "Exponential"
    Matern = "Matern"
    Linear = "Linear"


class KrigingMethod(Enum):
    Ordinary = "Ordinary"
    Simple = "Simple"
