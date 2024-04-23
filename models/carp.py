# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import math


class CARP(nn.Module):
    
    def __init__(self, opt, uori='user'):
        super(CARP, self).__init__()
        self.opt = opt
        self.num_fea = 1 # DOC

        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300

        self.id_user_emb = nn.Embedding(opt.user_num, opt.id_emb_size)
        self.id_item_emb = nn.Embedding(opt.item_num, opt.id_emb_size)

        self.user_viewpoint = ViewPoint(opt)
        self.item_viewpoint = ViewPoint(opt)

        self.logic_unit = nn.Linear(opt.id_emb_size * 2, opt.id_emb_size, bias=False)


    
    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas

        user_doc = self.user_word_embs(user_doc) # o_shape (bs, doc_len, word_dim)
        item_doc = self.item_word_embs(item_doc)

        user_ids = self.id_user_emb(uids)
        item_ids = self.id_item_emb(iids)

        # === PADDING TECNIQUE === # 
        pad_size = (self.opt.kernel_size - 1) // 2
        u_fea = F.pad(user_doc, (0, 0, pad_size, pad_size), 'constant', 0)
        i_fea = F.pad(item_doc, (0, 0, pad_size, pad_size), 'constant', 0)

        u_viewpoints = self.user_viewpoint(u_fea, user_ids)
        i_viewpoints = self.item_viewpoint(i_fea, item_ids)
        
        # u_fea = F.relu(self.user_cnn(u_fea.unsqueeze(1))).squeeze(3)  # o_shape (bs, filters, doc_len) 
        # i_fea = F.relu(self.item_cnn(i_fea.unsqueeze(1))).squeeze(3)  # 

        u_att = torch.mean(u_viewpoints, dim=1)
        i_att = torch.mean(i_viewpoints, dim=1)
        u_att = F.softmax(u_viewpoints * u_att.unsqueeze(1), dim=1)
        i_att = F.softmax(i_viewpoints * i_att.unsqueeze(1), dim=1)

        u_fea = torch.sum(u_viewpoints * u_att, 1)
        i_fea = torch.sum(i_viewpoints * i_att, 1)

        gated_unit = torch.cat([(u_fea - i_fea), (u_fea * i_fea)], dim=1)

        gated_unit = self.logic_unit(gated_unit)
        coupling_coeff = F.softmax(gated_unit, dim=1)

        sentiment_capsule = torch.sum(gated_unit * coupling_coeff, dim=1)

        routing = (math.pow(sentiment_capsule.size(0), 2)) / (math.pow(sentiment_capsule.size(0), 2) + 1) 
        routing = (sentiment_capsule / sentiment_capsule.size(0)) * routing


        # return user_vector, item_vector
        return u_fea, i_fea

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
    

class ViewPoint(nn.Module):
    def __init__(self, opt):
        super(ViewPoint, self).__init__()

        self.opt = opt

        self.cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))

        self.review_linear = nn.Linear(opt.filters_num, 1)
        self.id_linear = nn.Linear(opt.id_emb_size, opt.filters_num, bias=False)

        self.global_viewpoint = nn.Linear(opt.filters_num, opt.id_emb_size, bias=False)


    def forward(self, fea, id_fea):
        fea = F.relu(self.cnn(fea.unsqueeze(1)).squeeze(3)).transpose(2, 1)
        viewpoint = F.sigmoid(self.review_linear(fea) + self.id_linear(id_fea).unsqueeze(1))
        context = fea * viewpoint

        viewpoint = self.global_viewpoint(context)

        return viewpoint
    
    def reset_param(self):
        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)

        for x in [self.review_linear,
                  self.id_linear,
                  self.global_viewpoint]:
            nn.init.uniform_(x.weight, -0.1, 0.1)
            nn.init.constant_(x.bias, 0.1)        
