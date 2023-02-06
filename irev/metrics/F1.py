import numpy as np

from irev.metrics.BaseMetric import BaseMetric
from irev.metrics.Precision import Precision
from irev.metrics.Recall import Recall

class F1(BaseMetric):

    def calculate(prediction_matrix: np.ndarray, train_dataset: np.ndarray, test_dataset: np.ndarray, **kwargs) -> np.float64:
        threshold = float(kwargs.get("threshold", 3.5))
        ks = kwargs.get("k")

        if type(ks) is not list: ks = [ ks ]
        
        r = Recall.calculate(prediction_matrix, train_dataset, test_dataset, k=ks, threshold=threshold)
        p = Precision.calculate(prediction_matrix, train_dataset, test_dataset, k=ks, threshold=threshold)

        output = dict()

        for k in ks:
            k = int(k)
            output[k] = np.float128(2) * (p[k] * r[k]) / (p[k] + r[k])
            # output[k] = np.multiply([ 2 ], [ (p[k] * r[k]) / (p[k] + r[k]) ])

        return output
