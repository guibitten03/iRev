import numpy as np
from .relevance import relevance_metric

def dcg(relevance):
    """
    Calculate discounted cumulative gain.
    """
    
    if relevance is None or len(relevance) < 1:
        return 0.0
    
    rel = np.asarray(relevance)
    p = len(rel)

    log2i = np.log2(range(2, p + 1))
    return rel[0] + (rel[1:] / log2i).sum()

def idcg(relevance):
    """
    Calculate ideal discounted cumulative gain (maximum possible DCG).
    """

    if relevance is None or len(relevance) < 1:
        return 0.0
    
    rel = np.asarray(relevance).copy()
    rel.sort()

    return dcg(rel[::-1])


def ndcg_metric(ground_truth, predict, nranks):
    """
    Calculate normalized discounted cumulative gain.
    """

    relevance = relevance_metric(ground_truth, predict, nranks)    

    if (nranks < 1):
        raise Exception('nranks < 1')
    
    rel = np.asarray(relevance)
    pad = max(0, nranks - len(rel))

    rel = np.pad(rel, (0, pad), 'constant')

    rel = rel[0:min(nranks, len(rel))]

    ideal_dcg = idcg(rel)
    if ideal_dcg == 0: return 0.0

    return dcg(rel) / ideal_dcg