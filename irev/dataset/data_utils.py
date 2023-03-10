import re
import numpy as np


def dataset_to_rating_matrix(dataset: np.ndarray, U: int, I: int) -> np.ndarray:
    matrix = np.zeros(shape=(U, I))

    for row in dataset:
        uid = row[0]
        iid = row[1]
        rating = row[2]

        matrix[uid][iid] = rating

    return matrix
