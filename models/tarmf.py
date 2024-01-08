# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import pandas as pd

from .pmf import ProbabilisticMatrixFatorization


class TARMF(nn.Module):
    '''
    WWW 2018 TARMF
    '''
    def __init__(self, opt):
        super(TARMF, self).__init__()

        self.num_fea = 2
        self.opt = opt

        self.user_embeddings = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_embeddings = nn.Embedding(opt.vocab_size, opt.word_dim) # (bs, 500, 300)

        self.u_forward_gru = nn.GRU(input_size=opt.word_dim, hidden_size=1, bidirectional=True,batch_first=True)
        self.i_forward_gru = nn.GRU(input_size=opt.word_dim, hidden_size=1, bidirectional=True,batch_first=True)

        self.user_topic_att = nn.Sequential(
                            nn.Linear(2, 1),
                            nn.Tanh() 
        )
        self.item_topic_att = nn.Sequential(
                            nn.Linear(2, 1),
                            nn.Tanh()
        )

        self.user_proj = nn.Sequential(
                            nn.Linear(2, opt.id_emb_size),
                            nn.Tanh()
        )

        self.item_proj = nn.Sequential(
                            nn.Linear(2, opt.id_emb_size),
                            nn.Tanh()
        )

        train = pd.read_csv(f"dataset/.data/{self.opt.dataset}_{self.opt.emb_opt}/train/Train.csv")
        self.fit_pmf(train)

        self.reset_para()

    
    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas

        u_emb = self.user_embeddings(user_doc)
        i_emb = self.item_embeddings(item_doc)

        '''
            SEQUENCE ENCODING LAYER
            Eq 1 - 4
        '''
        u_emb, u_hn = self.u_forward_gru(u_emb)
        i_emb, i_hn = self.i_forward_gru(i_emb)

        '''
            TOPICAL ATTENTION LAYER
            Eq 5 - 7
        '''
        u_att = F.softmax(self.user_topic_att(u_emb), 1)
        i_att = F.softmax(self.item_topic_att(i_emb), 1)

        u_emb = u_emb * u_att
        i_emb = i_emb * i_att

        user_fea = torch.sum(u_emb, 1)
        item_fea = torch.sum(i_emb, 1)

        user_fea = self.user_proj(user_fea)
        item_fea = self.item_proj(item_fea)

        self.pmf_model.user_features = self.pmf_model.user_features.to('cuda')
        self.pmf_model.item_features = self.pmf_model.item_features.to('cuda')
        u_r_fea = self.pmf_model.user_features[uids]
        i_r_fea = self.pmf_model.item_features[iids]

        user_fea = torch.cat([user_fea, u_r_fea], dim=1)
        item_fea = torch.cat([item_fea, i_r_fea], dim=1)

        return user_fea, item_fea
    

    def reset_para(self):
        for x in [self.user_topic_att,
                  self.item_topic_att,
                  self.user_proj,
                  self.item_proj]:
            nn.init.uniform_(x[0].weight, -0.1, 0.1)
            nn.init.constant_(x[0].bias, 0.1)


    def fit_pmf(self, dataset):

        self.pmf_model = ProbabilisticMatrixFatorization(dataset)
        self.pmf_model.fit()