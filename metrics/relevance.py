import numpy as np

def relevance_metric(ground_truth, predict, nranks):

    ground_truth = np.asarray(ground_truth)
    predict = np.asarray(predict)

    relevance_score = np.asarray(range(1, len(ground_truth) + 1)[::-1])

    relevance = []
    for item in ground_truth: # Para todos os itens que são relevantes para meu usuário
        if item in predict[:len(ground_truth)]:
            relevance.append(relevance_score[np.argwhere(ground_truth == item)][0][0] / len(ground_truth))
        else:
            relevance.append(0)


    return relevance

