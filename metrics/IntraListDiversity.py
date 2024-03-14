import logging
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


def intra_list_dissimilarity(
    predicted: List[list], feature_df: pd.DataFrame, k: int
) -> float:
    """
    Computes the average intra-list dissimilarity of all recommendations.
    This metric can be used to measure diversity of the list of recommended items.
    Args:
        predicted : a list of lists
            Ordered predictions
            Example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        feature_df: dataframe
            A dataframe with one hot encoded or latent features.
            The dataframe should be indexed by the id used in the recommendations.
        k : integer
            The number of items to be considered at the ranking
    Returns:
        The average intra-list dissimilarity for recommendations.
    """
    # asserts
    assert k > 0, f"Value k={k} is not acceptable."
    assert len(predicted[0]) > 0, "There is not any prediction."
    # fill df
    feature_df = feature_df.fillna(0)

    # get all items recommended at least once that have some features
    recs = {i for pred in predicted for i in pred}
    recs = list(recs.intersection(feature_df.index))
    # get their features
    recs_content = feature_df.loc[recs]
    recs_content = recs_content.dropna()

    # save a map for each item-id
    items_map = dict(zip(recs_content.index, np.arange(0, recs_content.shape[0])))
    # create the sparse matrix
    recs_content = sp.csr_matrix(np.array(recs_content.values, dtype=int))
    # calculate similarity scores for all items recommended
    similarity = cosine_similarity(X=recs_content, dense_output=False)
    exceptions = []

    def get_list_dissimilarity(predictions: list) -> float:

        if len(predictions) > k:
            predictions = predictions[:k]

        ild_single_user = []
        # get similarities
        for pos, i in enumerate(predictions):
            if i in items_map:
                i_index = items_map[i]
                for j in predictions[pos + 1:]:
                    if j in items_map:
                        j_index = items_map[j]
                        ild_single_user.append(1.0-similarity[i_index, j_index])
            else:
                exceptions.append(i)

        return np.mean(ild_single_user)

    # Running metric
    results = list(map(get_list_dissimilarity, predicted))
    if len(exceptions) > 0:
        logging.warning(f"The podcasts {set(exceptions)} do not have any categorical feature.")

    return np.mean(results)