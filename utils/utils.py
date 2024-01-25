import time
import numpy as np
import torch

# METRICS IMPORT
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ndcg_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

def calculate_metrics(y_true, y_pred):
    y_true = torch.nan_to_num(y_true, nan=3.0)
    y_pred = torch.nan_to_num(y_pred, nan=3.0)
    rmse = mean_squared_error(y_true.cpu(), y_pred.cpu(), squared=False)
    # ndcg = ndcg_score(np.array(y_true.cpu()).reshape(1, -1), np.array(y_pred.cpu()).reshape(1, -1), k=4)
    precision = precision_score(np.around(y_true.cpu()), np.around(y_pred.cpu()), average="macro", zero_division=np.nan)
    recall = recall_score(np.around(y_true.cpu()), np.around(y_pred.cpu()), average="macro", zero_division=np.nan)

    return rmse, precision, recall