import numpy as np

from irev.algorithms.BaseAlgorithm import BaseAlgorithm


class RMG(BaseAlgorithm):

    def __init__(self):
        pass

    def fit(self, data: np.ndarray, U: int, I: int) -> np.ndarray:
        return np.random.rand(U, I) * 5
