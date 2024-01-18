import pandas as pd
from pmf import ProbabilisticMatrixFatorization
import fire
import numpy as np


def train(**kwargs):
    train = pd.read_csv(f"../dataset/.data/{kwargs['dataset']}_{kwargs['emb_opt']}/train/Train.csv")

    pmf_model = ProbabilisticMatrixFatorization(train)
    pmf_model.fit()

    user_features = pmf_model.user_features.cpu().detach().numpy()
    item_features = pmf_model.item_features.cpu().detach().numpy()

    with open(f"../checkpoints/user_features_pmf_{kwargs['dataset']}_{kwargs['emb_opt']}.npy", 'wb') as f:
        np.save(f, user_features)

    with open(f"../checkpoints/item_features_pmf_{kwargs['dataset']}_{kwargs['emb_opt']}.npy", 'wb') as f:
        np.save(f, item_features)


if __name__ == "__main__":
    fire.Fire()
        
        

