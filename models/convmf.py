# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

from .pmf import ProbabilisticMatrixFatorization


class ConvMF(nn.Module):
    
    def __init__(self, opt, uori='user'):
        super(ConvMF, self).__init__()
        self.opt = opt
        self.num_fea = 1 # DOC

        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300

        self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))

        self.item_fc_linear = nn.Linear(opt.filters_num, opt.fc_dim)
        self.dropout = nn.Dropout(self.opt.drop_out)
        
        train = pd.read_csv(f"dataset/.data/{self.opt.dataset}_{self.opt.emb_opt}/train/Train.csv")
        self.fit_pmf(train)

        self.reset_para()   

    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas


        item_doc = self.item_word_embs(item_doc)

        i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)
        i_fea = self.dropout(self.item_fc_linear(i_fea))

        self.pmf_model.user_features = self.pmf_model.user_features.to('cuda')

        u_fea = self.pmf_model.user_features[uids]

        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)


    def reset_para(self):

        nn.init.xavier_normal_(self.item_cnn.weight)

        nn.init.uniform_(self.item_fc_linear.weight, -0.1, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.item_word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)


    def fit_pmf(self, dataset):

        self.pmf_model = ProbabilisticMatrixFatorization(dataset)
        self.pmf_model.fit()
