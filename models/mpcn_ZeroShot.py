# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MPCN_ZeroShot(nn.Module):
    
    def __init__(self, opt, head=3):
        
        super(MPCN_ZeroShot, self).__init__()

        self.opt = opt
        self.num_fea = 1  
        self.head = head

        self.fc_g1 = nn.Linear(768, 768)
        self.fc_g2 = nn.Linear(768, 768)

        self.review_coatt = nn.ModuleList([Co_Attention(768, gumbel=True, pooling='max') for _ in range(head)])
        self.word_coatt = nn.ModuleList([Co_Attention(768, gumbel=False, pooling='avg') for _ in range(head)])

        self.u_fc = self.fc_layer()
        self.i_fc = self.fc_layer()

        self.drop_out = nn.Dropout(opt.drop_out)
        self.reset_para()

    def fc_layer(self):
        return nn.Sequential(
            nn.Linear(768 * self.head, 768),
            nn.ReLU(),
            nn.Linear(768, self.opt.id_emb_size)
        )

    def forward(self, datas):
        
        user_reviews, item_reviews, _, _, _, _, _, _ = datas

        u_reviews = self.review_gate(user_reviews)
        i_reviews = self.review_gate(item_reviews)

        u_fea = []
        i_fea = []
        for i in range(self.head):
            r_coatt = self.review_coatt[i]
            w_coatt = self.word_coatt[i]

            p_u, p_i = r_coatt(u_reviews, i_reviews)        
            p_u, p_i = w_coatt(u_reviews, i_reviews)                
            u_w_fea = u_reviews.permute(0, 2, 1).bmm(p_u).squeeze(2)
            i_w_fea = i_reviews.permute(0, 2, 1).bmm(p_i).squeeze(2)
            u_fea.append(u_w_fea)
            i_fea.append(i_w_fea)

        u_fea = torch.cat(u_fea, 1)
        i_fea = torch.cat(i_fea, 1)

        u_fea = self.drop_out(self.u_fc(u_fea))
        i_fea = self.drop_out(self.i_fc(i_fea))

        return torch.stack([u_fea], 1), torch.stack([i_fea], 1)

    def review_gate(self, reviews):
        return torch.sigmoid(self.fc_g1(reviews)) * torch.tanh(self.fc_g2(reviews))

    def reset_para(self):
        for fc in [self.fc_g1, self.fc_g2, self.u_fc[0], self.u_fc[-1], self.i_fc[0], self.i_fc[-1]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)


class Co_Attention(nn.Module):
    
    def __init__(self, dim, gumbel, pooling):
        super(Co_Attention, self).__init__()
        self.gumbel = gumbel
        self.pooling = pooling
        self.M = nn.Parameter(torch.randn(dim, dim))
        self.fc_u = nn.Linear(dim, dim)
        self.fc_i = nn.Linear(dim, dim)

        self.reset_para()

    def reset_para(self):
        nn.init.xavier_uniform_(self.M, gain=1)
        nn.init.uniform_(self.fc_u.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_u.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_i.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_i.bias, -0.1, 0.1)

    def forward(self, u_fea, i_fea):
       
        u = self.fc_u(u_fea)
        i = self.fc_i(i_fea)
        S = u.matmul(self.M).bmm(i.permute(0, 2, 1))  # B * L1 * L2 Eq(2/10), we transport item instead user
        if self.pooling == 'max':
            u_score = S.max(2)[0]  # B * L1
            i_score = S.max(1)[0]  # B * L2
        else:
            u_score = S.mean(2)  # B * L1
            i_score = S.mean(1)  # B * L2
        if self.gumbel:
            p_u = F.gumbel_softmax(u_score, hard=True, dim=1)
            p_i = F.gumbel_softmax(i_score, hard=True, dim=1)
        else:
            p_u = F.softmax(u_score, dim=1)
            p_i = F.softmax(i_score, dim=1)
        return p_u.unsqueeze(2), p_i.unsqueeze(2)
