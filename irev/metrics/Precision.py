from typing import Dict, SupportsFloat

import numpy as np

from irev.enviroment import utils
from irev.metrics import BaseMetric


class Precision(BaseMetric):

    def calculate(prediction_matrix: np.ndarray, train_dataset: np.ndarray, test_dataset: np.ndarray, **kwargs) -> Dict[int, SupportsFloat]:
        U, I = prediction_matrix.shape

        test_matrix = utils.dataset_to_rating_matrix(test_dataset, U, I)

        # Set to 0 the items in the prediction matrix that are already present in the test dataset.
        prediction_matrix = (utils.dataset_to_rating_matrix(train_dataset, U, I) == 0) * prediction_matrix

        output = dict()

        ks = kwargs.get("k")
        threshold = float(kwargs.get("threshold", 3.5))

        if type(ks) is not list: ks = [ ks ]

        for k in ks:
            k = int(k)

            sum_precision = 0

            for user_predictions, actual_ratings in zip(prediction_matrix, test_matrix):
                top_k_recommendations = np.argsort(user_predictions)[::-1][:k]

                ratings_recommended_items = user_predictions[top_k_recommendations]
                true_ratings_recommended_items = actual_ratings[top_k_recommendations]

                recommended = ratings_recommended_items >= threshold
                relevant = true_ratings_recommended_items >= threshold

                recommended_and_relevant = np.sum(recommended & relevant)
                recommended_and_not_relevant = np.sum(recommended & np.invert(relevant))

                user_precision = recommended_and_relevant / (recommended_and_relevant + recommended_and_not_relevant)

                sum_precision = sum_precision + user_precision

            output[k] = sum_precision / U

        return output
