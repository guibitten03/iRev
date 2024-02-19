# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DeepCoNNBERT(nn.Module):
    '''
    deep conn 2017
    '''
    def __init__(self, opt, uori='user'):
        super(DeepCoNNBERT, self).__init__()
        self.opt = opt
        self.num_fea = 1 # DOC
        

        self.user_cnn = nn.Conv1d(1, opt.filters_num, opt.kernel_size)
        self.item_cnn = nn.Conv1d(1, opt.filters_num, opt.kernel_size)

        self.user_fc_linear = nn.Linear(opt.u_bert_doc_size, opt.fc_dim)
        self.item_fc_linear = nn.Linear(opt.i_bert_doc_size, opt.fc_dim)
        self.dropout = nn.Dropout(self.opt.drop_out)

        self.reset_para()   

    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas

        # u_fea = F.relu(self.user_cnn(user_doc.unsqueeze(1)))  # .permute(0, 2, 1)
        # i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1)))


        # u_fea = F.max_pool1d(u_fea, u_fea.size(2)).squeeze(2)
        # i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)
        # u_fea = self.dropout(self.user_fc_linear(u_fea))
        # i_fea = self.dropout(self.item_fc_linear(i_fea))
        u_fea = self.dropout(self.user_fc_linear(user_doc))
        i_fea = self.dropout(self.item_fc_linear(item_doc))


        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)

    def reset_para(self):

        for cnn in [self.user_cnn, self.item_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for fc in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)
