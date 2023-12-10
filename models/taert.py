# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TAERT(nn.Module):
    
    def __init__(self, opt, uori='user'):
        super(TAERT, self).__init__()

        self.opt = opt
        self.num_fea = 1

        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')


    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, _, _, _, _ = datas

        user_features = self.user_net(user_reviews, uids)
        item_features = self.item_net(item_reviews, iids)

        return user_features, item_features


class Net(nn.Module):
    def __init__(self, opt, uori='user'):
        super(Net, self).__init__()

        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        self.word_emb = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.id_emb = nn.Embedding(id_num, self.opt.id_emb_size)

        # Temporal Convolutional Network
        self.tcn = TemporalConvolutionalNetwork(opt)

        self.review_att = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
        self.word_att = nn.Linear(self.opt.id_emb_size, 1, bias=False)

        self.id_linear = nn.Linear(self.opt.id_emb_size, self.opt.id_emb_size, bias=False)
        self.review_linear = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)
        self.attention_linear = nn.Linear(self.opt.id_emb_size, 1)

        self.fc = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)

    def forward(self, reviews, ids):
        
        ids_emb = self.id_emb(ids)

        reviews = self.word_emb(reviews)
        bs, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)

        # --- TCN  --- #
        reviews_fea = self.tcn(reviews.unsqueeze(1)).squeeze(3)
        reviews_fea = F.max_pool1d(reviews_fea, reviews_fea.size(2)).squeeze(2)
        reviews_fea = reviews_fea.view(-1, r_num, reviews_fea.size(1))

        # --- Word Level Attention ---- #
        att_weight = F.tanh(self.review_att(reviews_fea))
        att_weight = F.softmax(self.word_att(att_weight), 1)

        reviews_fea = reviews_fea * att_weight
        reviews_fea = reviews_fea.sum(1)

        # --- Review Level Attention ---- #
        # reviews_att = F.relu(self.review_linear(reviews_fea) + self.id_linear(ids_emb))
        # reviews_att = F.softmax(self.attention_linear(reviews_att), 1)

        # # ---  --- #
        # reviews_fea = self.fc(reviews_fea * reviews_att)
        reviews_fea = self.fc(reviews_fea)

        return reviews_fea



class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, opt):
        super(TemporalConvolutionalNetwork, self).__init__()

        self.tcn = nn.Sequential(
            # Ideal is to be padding='same'
            nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), dilation=(2 ** 0)),
            nn.BatchNorm2d(opt.filters_num),
            nn.ReLU(),
            nn.Dropout(opt.drop_out),
            nn.Conv2d(opt.filters_num, opt.filters_num, (opt.kernel_size, 1), dilation=(2 ** 0)),
            nn.BatchNorm2d(opt.filters_num),
            nn.ReLU(),
            nn.Dropout(opt.drop_out),
        )

    def forward(self, x):
        features = self.tcn(x)
        return features