# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import pandas as pd


class ProbabilisticMatrixFatorization():
    def __init__(self, iteractions):

        rating_matrix = pd.pivot_table(iteractions, values="ratings", index="user_id", columns="item_id")
        self.n_users, self.n_items = rating_matrix.shape

        min_rating, max_rating = iteractions['ratings'].min(), iteractions['ratings'].max()

        rating_matrix[rating_matrix.isna()] = -1

        rating_matrix = (rating_matrix - min_rating) / (max_rating - min_rating)


        self.rating_matrix = torch.FloatTensor(rating_matrix.values)

        latent_vectors = 32

        self.user_features = torch.randn(self.n_users, latent_vectors, requires_grad=True)
        self.user_features.data.mul_(0.01)
        self.item_features = torch.randn(self.n_items, latent_vectors, requires_grad=True)
        self.item_features.data.mul_(0.01)

        self.rating_matrix.cuda()
        self.user_features.cuda()
        self.item_features.cuda()

    
    def fit(self):

        optimizer = torch.optim.Adam([self.user_features, self.item_features], lr=0.01)

        for step, epoch in enumerate(range(1000)):
            predict = torch.matmul(self.user_features, self.item_features.T)

            loss = torch.sum((self.rating_matrix - predict) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step}, {loss:.3f}")