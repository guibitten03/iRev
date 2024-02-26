# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

import pickle as pkl
import joblib

from .pmf import ProbabilisticMatrixFatorization


class ConvMF_FineTunning(nn.Module):
    
    def __init__(self, opt, uori='user'):
        super(ConvMF_FineTunning, self).__init__()
        self.opt = opt
        self.num_fea = 1 # DOC

        self.item_cnn = nn.Conv2d(1, opt.filters_num, (1, 768))

        self.item_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        self.dropout = nn.Dropout(self.opt.drop_out)
        
        self.user_features = torch.Tensor(np.load(f"checkpoints/user_features_pmf_{opt.dataset}_{opt.emb_opt}.npy")).to('cuda')
        self.item_features = torch.Tensor(np.load(f"checkpoints/item_features_pmf_{opt.dataset}_{opt.emb_opt}.npy")).to('cuda')

        self.reset_para()   

    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas


        i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)
        i_fea = self.dropout(self.item_fc_linear(i_fea))

        u_fea = self.user_features[uids]

        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)


    def reset_para(self):

        nn.init.xavier_normal_(self.item_cnn.weight)
        nn.init.constant_(self.item_cnn.bias, 0.1)

        nn.init.uniform_(self.item_fc_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.item_fc_linear.bias, 0.1)


    # def fit_pmf(self, dataset):

    #     self.pmf_model = ProbabilisticMatrixFatorization(dataset)
    #     self.pmf_model.fit()
