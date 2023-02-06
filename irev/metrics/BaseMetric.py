from abc import ABC, abstractmethod
from typing import Dict, SupportsFloat

import numpy as np


class BaseMetric(ABC):

    @abstractmethod
    def calculate(self, prediction_matrix: np.ndarray, train_dataset: np.ndarray, test_dataset: np.ndarray) -> SupportsFloat | Dict[int, SupportsFloat]:
        """
            :param prediction_matrix: The matrix of ratings outputed by the algorithms.
            :type prediction_matrix: np.ndarray, shape=(U, I)

            :param train_dataset: All the data informed to the model.
            :type train_dataset: np.ndarray, shape=(N, 5)

            :param test_dataset: The data to confirm the predictions.
            :type test_dataset: np.ndarray, shape=(N, 5)
        """
        raise Exception("Trying to call BaseMetric calculate method.")
