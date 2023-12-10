# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CARP(nn.Module):
    
    def __init__(self, opt, uori='user'):
        super(CARP, self).__init__()
        self.opt = opt
        self.num_fea = 1 # DOC

        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300

        self.user_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))

        self.user_selfattn = nn.MultiheadAttention(opt.filters_num, 10)
        self.item_selfattn = nn.MultiheadAttention(opt.filters_num, 10)

        self.view_shared = nn.Linear(opt.filters_num, opt.id_emb_size, bias=False)


    
    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas

        user_doc = self.user_word_embs(user_doc) # o_shape (bs, doc_len, word_dim)
        item_doc = self.item_word_embs(item_doc)

        # === PADDING TECNIQUE === # 
        pad_size = (self.opt.kernel_size - 1) // 2
        u_fea = F.pad(user_doc, (0, 0, pad_size, pad_size), 'constant', 0)
        i_fea = F.pad(item_doc, (0, 0, pad_size, pad_size), 'constant', 0)

        '''
            Context Encoding
        '''
        u_fea = F.relu(self.user_cnn(u_fea.unsqueeze(1))).squeeze(3)  # o_shape (bs, filters, doc_len) 
        i_fea = F.relu(self.item_cnn(i_fea.unsqueeze(1))).squeeze(3)  # 

        '''
            Self Attention
            Eq 1
        '''
        u_fea = torch.transpose(u_fea, 2, 1)
        i_fea = torch.transpose(i_fea, 2, 1)
        u_att_mask = F.sigmoid(self.user_selfattn(u_fea, u_fea, u_fea)[0]) # o_shape (bs, filters, doc_len)
        i_att_mask = F.sigmoid(self.item_selfattn(i_fea, i_fea, i_fea)[0]) 

        u_fea = u_fea * u_att_mask # o_shape (bs, filters, doc_len)
        i_fea = i_fea * i_att_mask

        # Viewpoint Shared Transform
        user_points = self.view_shared(u_fea) # o_shape (bs, doc_len, k=32)
        item_points = self.view_shared(i_fea)

        # Viewpoint Especific Context
        user_vector = torch.mean(user_points, dim=1)
        item_vector = torch.mean(item_points, dim=1)

        # Intra Attention
        user_vector = torch.sum(F.softmax(user_vector.unsqueeze(1) * user_points, dim=1), dim=1)
        item_vector = torch.sum(F.softmax(item_vector.unsqueeze(1) * item_points, dim=1), dim=1)

        '''
            Logic Unit Representation,
            Sentiment Capsules
            Eq: 3 
        '''
        # Neural Factorization Machine
        # logic_unit = torch.cat([user_vector, item_vector], dim=1)


        return user_vector, item_vector
        # return None, None