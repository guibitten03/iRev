import numpy as np

from irev.metrics import BaseMetric


class MSE(BaseMetric):

    def calculate(prediction_matrix: np.ndarray, train_dataset: np.ndarray, test_dataset: np.ndarray, **kwargs) -> np.float64:
        sum_deltas_squared = 0

        for interaction in test_dataset:
            uid = interaction[0]
            iid = interaction[1]

            real_rating = interaction[2]
            predicted_rating = prediction_matrix[uid][iid]

            diff_squared = (predicted_rating - real_rating) ** 2

            sum_deltas_squared = sum_deltas_squared + diff_squared

        return sum_deltas_squared / test_dataset.shape[0]
