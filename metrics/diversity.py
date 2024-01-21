import math

def diversity(true, pred):
    entropy = []

    mean = sum(pred) / len(pred)

    for t in true:
        prob = abs(t - mean)
        entropy.append(abs(-prob * math.log2(prob)))

    return 1 - (sum(entropy) / len(entropy))