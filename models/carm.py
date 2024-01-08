# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import pandas as pd

from .pmf import ProbabilisticMatrixFatorization


class CARM(nn.Module):

    def __init__(self, opt, uori='user'):
        super(CARM, self).__init__()
        self.opt = opt
        self.num_fea = 1 # DOC
        
        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300

        self.user_cnn = CNN(opt, uori='user')
        self.item_cnn = CNN(opt, uori='item')

        self.user_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        self.item_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        self.dropout = nn.Dropout(self.opt.drop_out)

        train = pd.read_csv(f"dataset/.data/{self.opt.dataset}_{self.opt.emb_opt}/train/Train.csv")
        self.fit_pmf(train)

        self.reset_para()   

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, _, _, _, _ = datas

        user_reviews = self.user_word_embs(user_reviews)
        item_reviews = self.item_word_embs(item_reviews)

        u_fea = self.user_cnn(user_reviews)
        i_fea = self.item_cnn(item_reviews)

        self.pmf_model.user_features = self.pmf_model.user_features.to('cuda')
        self.pmf_model.item_features = self.pmf_model.item_features.to('cuda')

        u_factors = self.pmf_model.user_features[uids]
        i_factors = self.pmf_model.item_features[iids]

        u_fea = u_fea + u_factors
        i_fea = i_fea + i_factors


        return u_fea, i_fea
        # return None, None

    def reset_para(self):

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v.cuda())
                self.item_word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)

        for x in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(x.weight, -0.1, 0.1)
            nn.init.constant_(x.bias, 0.1)


    def fit_pmf(self, dataset):

        self.pmf_model = ProbabilisticMatrixFatorization(dataset)
        self.pmf_model.fit()


class CNN(nn.Module):
    def __init__(self, opt, uori='user'):
        super(CNN, self).__init__()

        self.opt = opt

        if uori == 'user':
            self.r_num = opt.u_max_r
        else:
            self.r_num = opt.i_max_r

        self.conv = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.linear = nn.Sequential(
            nn.Linear(self.r_num * opt.filters_num, opt.id_emb_size),
            nn.Tanh(),
            nn.Linear(opt.id_emb_size, opt.id_emb_size),
            nn.Tanh()
        )

        self.reset_para()

    def forward(self, reviews):
        bs, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)

        fea = F.relu(self.conv(reviews.unsqueeze(1)).squeeze(3))
        fea = F.max_pool1d(fea, fea.size(2)).squeeze(2)
        fea = fea.view(-1, r_num, fea.size(1))

        fea = self.linear(fea.view(-1, fea.size(1) * fea.size(2)))

        return fea

    def reset_para(self):
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        for x in [self.linear[0], self.linear[2]]:
            nn.init.uniform_(x.weight, -0.1, 0.1)
            nn.init.constant_(x.bias, 0.1)