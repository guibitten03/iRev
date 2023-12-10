# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CARL(nn.Module):
    def __init__(self, opt):
        super(CARL, self).__init__()

        self.opt = opt
        self.num_fea = 3

        self.user_embedding = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_embedding = nn.Embedding(opt.vocab_size, opt.word_dim)

        self.u_emb = nn.Embedding(opt.user_num, opt.id_emb_size)
        self.i_emb = nn.Embedding(opt.item_num, opt.id_emb_size)

        self.user_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1,0))
        self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1,0))
        self.att_matrix = nn.Parameter(torch.Tensor(opt.filters_num, opt.filters_num), requires_grad=True)

        self.abstract_user_cnn = nn.Conv2d(opt.filters_num, 1, (1,1))
        self.abstract_item_cnn = nn.Conv2d(opt.filters_num, 1, (1,1))

        self.shared_mlp = nn.Linear(opt.doc_len, opt.id_emb_size)


    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas

        user_fea = self.user_embedding(user_doc)
        item_fea = self.item_embedding(item_doc)

        uids_emb = self.u_emb(uids)
        iids_emb = self.i_emb(iids)

        user_fea = F.relu(self.user_cnn(user_fea.unsqueeze(1)).squeeze(3))
        item_fea = F.relu(self.item_cnn(item_fea.unsqueeze(1)).squeeze(3))

        # --- Attentive Layer --- #
        relation_matrix = torch.matmul(user_fea.transpose(1,2), self.att_matrix)
        relation_matrix = F.tanh(torch.matmul(relation_matrix, item_fea))

        user_att = F.softmax(torch.mean(relation_matrix, 1), dim=1).unsqueeze(1)
        item_att = F.softmax(torch.mean(relation_matrix, 2), dim=1).unsqueeze(1)

        # --- Abstracting Layer  --- #
        user_fea = user_fea * user_att
        item_fea = item_fea * item_att

        user_fea = self.abstract_user_cnn(user_fea.unsqueeze(3)).squeeze(3)
        item_fea = self.abstract_item_cnn(item_fea.unsqueeze(3)).squeeze(3)

        user_fea = self.shared_mlp(user_fea).squeeze(1)
        item_fea = self.shared_mlp(item_fea).squeeze(1)

        abstratic_fea = torch.stack([(user_fea * item_fea), user_fea, item_fea], 1)
        interaction_fea = torch.stack([(uids_emb * iids_emb), uids_emb, iids_emb], 1)

        return abstratic_fea, interaction_fea
