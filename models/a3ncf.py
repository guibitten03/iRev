# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class A3NCF(nn.Module):
    
    def __init__(self, opt, uori='user'):
        super(A3NCF, self).__init__()
        self.opt = opt
        self.num_fea = 1 

        
        self.user_emb = nn.Embedding(opt.user_num, opt.id_emb_size)
        self.item_emb = nn.Embedding(opt.item_num, opt.id_emb_size)

        self.user_latent = nn.Flatten()
        self.item_latent = nn.Flatten()

        self.user_att1 = nn.Sequential(
            nn.Linear(opt.id_emb_size, opt.id_emb_size),
            nn.ReLU()
        )

        self.item_att1 = nn.Sequential(
            nn.Linear(opt.id_emb_size, opt.id_emb_size),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear((opt.id_emb_size*2) + (opt.id_emb_size*2), opt.id_emb_size),
            nn.Softmax(dim=0)
        )

        self.prediction_mlp = nn.Sequential(
            nn.Linear(opt.id_emb_size, opt.id_emb_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_out),
            nn.Linear(opt.id_emb_size, 1)
        )
        

        self.reset_para()   

    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas

        user_fea = self.user_emb(uids)
        item_fea = self.item_emb(iids)

        user_flat = self.user_latent(user_fea)
        item_flat = self.item_latent(item_fea)

        user_latent = user_doc + user_flat
        item_latent = item_doc + item_flat

        user_latent = self.user_att1(user_latent)
        item_latent = self.item_att1(item_latent)

        ui_latent = torch.concat([user_doc, item_doc, user_latent, item_latent], dim=1)
        attention_fea = self.attention(ui_latent)
        condense_vec = torch.matmul(user_flat, item_flat.T)

        predict_vec = torch.matmul(condense_vec, attention_fea)

        pred = self.prediction_mlp(predict_vec)

        return pred.squeeze(1), None

    def reset_para(self):

        for emb in [self.user_emb, self.item_emb]:
            nn.init.uniform_(emb.weight)

        for fc in [self.user_att1[0], 
                   self.item_att1[0],
                   self.attention[0],
                   self.prediction_mlp[0],
                   self.prediction_mlp[3]]:
            
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)
