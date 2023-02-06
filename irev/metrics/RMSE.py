import numpy as np

from irev.metrics import MSE


class RMSE(MSE):
    def calculate(prediction_matrix: np.ndarray, train_dataset: np.ndarray, test_dataset: np.ndarray, **kwargs):
        return np.sqrt(MSE.calculate(prediction_matrix, train_dataset, test_dataset))
