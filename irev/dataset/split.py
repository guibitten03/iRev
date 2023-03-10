import numpy as np
import pandas as pd

## Separar treino em treino e validação - 0.5 para cada

def user_based_split(df: pd.DataFrame, train_size: float):
    df = df.sort_values(by=[ "user_id", "timestamp" ])

    uids = df["user_id"].unique()

    train_data, test_data = list(), list()

    for uid in uids:
        user_data = df[(df["user_id"] == uid)]
        user_num_interactions = len(user_data)

        train_amount = round(train_size * user_num_interactions)

        user_train_data = user_data[:train_amount]
        user_test_data = user_data[train_amount:]

        train_data.append(user_train_data.values)
        test_data.append(user_test_data.values)

    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    return train_data, test_data