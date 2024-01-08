# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class NRPA(nn.Module):
    def __init__(self, opt):
        super(NRPA, self).__init__()

        self.num_fea = 1

        self.user_word_emb = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_word_emb = nn.Embedding(opt.vocab_size, opt.word_dim)

        self.user_id_emb = nn.Embedding(opt.user_num, opt.id_emb_size)
        self.item_id_emb = nn.Embedding(opt.item_num, opt.id_emb_size)

        self.user_conv = Conv(opt)
        self.item_conv = Conv(opt)



    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, _, _, _, _ = datas

        u_fea = self.user_word_emb(user_reviews)
        i_fea = self.item_word_emb(item_reviews)

        u_id_fea = self.user_id_emb(uids)
        i_id_fea = self.item_id_emb(iids)

        u_fea = self.user_conv(u_fea)
        i_fea = self.item_conv(i_fea)

        print(u_fea.shape)

        return None, None


class Conv(nn.Module):
    def __init__(self, opt):
        super(Conv, self).__init__()

        self.opt = opt

        self.conv = nn.Conv2d(1, opt.filters_num, kernel_size=(opt.kernel_size, opt.word_dim))

    
    def forward(self, reviews):
        bs, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)

        fea = F.relu(self.conv(reviews.unsqueeze(1)).squeeze(3))
        

        return fea