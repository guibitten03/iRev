from abc import ABC, abstractmethod

import numpy as np


class BaseAlgorithm(ABC):

    @abstractmethod
    def __init__(self):
        raise Exception("Trying to call BaseAlgorithm __init__ method.")

    @abstractmethod
    def fit(self, data: np.ndarray, U: int, I: int) -> np.ndarray:
        """
            :param data: The data to train the model.
            :type data: np.ndarray, shape=(test_size * dataset_size, 5)

            :param U: The total amount of users in the dataset.
            :type U: int

            :param I: The total amount of items in the dataset.
            :type I: int

            :return:  Matrix prediction with shape U x I.
            :rtype: np.ndarray, shape=(U, I)
        """
        raise Exception("Trying to call BaseAlgorithm fit method.")
