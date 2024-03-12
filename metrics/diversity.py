import math

def diversity(itens, pred_cat):
    entropy = []

    for item, cat in zip(itens, pred_cat):
        for alt, cat_alt in zip(itens, pred_cat):
            if item == alt: continue
            if cat == cat_alt:
                entropy.append(1)
            else:
                entropy.append(0)

    entropy_sum = sum(entropy)

    if entropy_sum == 0:
        return 1
    
    entropy = math.log2(sum(entropy))

    return 1 - (entropy / len(itens))
