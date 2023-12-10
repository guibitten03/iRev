# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ALFM(nn.Module):

    def __init__(self, opt):
        super(ALFM, self).__init__()

        self.opt = opt

        self.num_fea = 1   # Document

        self.user_embedding = nn.Embedding(
					opt.user_num, opt.id_emb_size)
        self.item_embedding = nn.Embedding(
					opt.item_num, opt.id_emb_size)

        self.user_fusion = nn.Sequential(
			nn.Linear(opt.id_emb_size, opt.id_emb_size),
			nn.ReLU())
        self.item_fusion = nn.Sequential(
			nn.Linear(opt.id_emb_size, opt.id_emb_size),
			nn.ReLU())

        self.att_layer1 = nn.Sequential(
			nn.Linear(2 * opt.id_emb_size, 1),
			nn.ReLU())
        self.att_layer2 = nn.Linear(1, opt.id_emb_size, bias=False)

        self.rating_predict = nn.Sequential(
			nn.Linear(opt.id_emb_size, opt.id_emb_size),
			nn.ReLU(),
			nn.Dropout(p=opt.drop_out),
			# nn.Linear(opt.id_emb_size, opt.id_emb_size),
			# nn.ReLU(),
			# nn.Dropout(p=self.dropout),
			nn.Linear(opt.id_emb_size, 1))



    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas

		########################## INPUT #########################
        user_id_embed = self.user_embedding(uids)
        item_id_embed = self.item_embedding(iids)
		
		###################### FEATURE FUSION ####################
        user_embed = user_id_embed + user_doc
        item_embed = item_id_embed + item_doc
        user_embed = self.user_fusion(user_embed)
        item_embed = self.item_fusion(item_embed)

		################## ATTENTIVE INTERACTION ################
        feature_all = torch.cat((
					user_embed, item_embed), dim=-1)
        att_weights = self.att_layer2(self.att_layer1(feature_all))
        att_weights = F.softmax(att_weights, dim=-1)

		#################### RATING PREDICTION ###################
        interact = att_weights * user_embed * item_embed
        prediction = self.rating_predict(interact)

        # prediction = 5 * torch.sigmoid(prediction)

        return prediction.squeeze(1), None

    
    def reset_para(self):

        for emb in [self.user_embedding, self.item_embedding]:
            nn.init.xavier_uniform_(emb.weight)

        for fc in [self.user_fusion,
                   self.item_fusion,
                   self.att_layer1,
                   self.att_layer2,
                   self.rating_predict[0],
                   self.rating_predict[3]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)
    