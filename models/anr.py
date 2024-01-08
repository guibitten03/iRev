# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ANR(nn.Module):
    
    def __init__(self, opt, uori='user'):
        super(ANR, self).__init__()
        self.opt = opt
        self.num_fea = 1 # DOC
        
        self.user_emb = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.user_emb.weight.requires_grad = False
        self.item_emb = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_emb.weight.requires_grad = False

        self.arl = AspectRepresentationLearning(opt)

        # self.aie = AspectImportanceEstimation(opt)
        
        # self.rp = RatingPrediction(opt)


    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas

        user_emb = self.user_emb(user_doc) # shape (batch_size, doc_len, word_dim)
        item_emb = self.item_emb(item_doc)

        '''
            Aspect Representation Learning
            Eq 1 - 4
        '''
        # user_emb = self.arl(user_emb)
        # item_emb = self.arl(item_emb)
        user_att, user_rep = self.arl(user_emb)
        item_att, item_rep = self.arl(item_emb)

        # '''
        #     Aspect Importance Estimation
        #     Eq 5 - 7
        # '''
        # user_coo_attn, item_coo_attn = self.aie(user_rep, item_rep)
        
        
        # '''
        #     Rating Importance Prediction
        #     Eq 8
        # '''
        # ratings = self.rp(user_rep, item_rep, user_coo_attn, item_coo_attn, uids, iids)


        return user_rep, item_rep
        # return ratings, None


class AspectRepresentationLearning(nn.Module):
    def __init__(self, opt):
        super(AspectRepresentationLearning, self).__init__()

        self.opt = opt

        self.aspect_projection = nn.Conv2d(1, 5, kernel_size=(3, opt.word_dim))
        self.shared_projection = nn.Linear(opt.doc_len, opt.doc_len, bias=False)


    def forward(self, doc):
        pad_size = (3 - 1) // 2
        doc = F.pad(doc, (0, 0, pad_size, pad_size), "constant", 0)
        aspects = self.aspect_projection(doc.unsqueeze(1)).squeeze(3)

        aspects_attention = F.softmax(self.shared_projection(aspects), dim=1)

        representation = torch.sum(aspects_attention * aspects, dim=1)

        return aspects_attention, representation


class AspectImportanceEstimation(nn.Module):
    def __init__(self, opt):
        super(AspectImportanceEstimation, self).__init__()

        self.opt = opt

        self.affinity_matrix = nn.Parameter(
            torch.Tensor(opt.doc_len, opt.doc_len), requires_grad=True
        )

        self.user_projection = nn.Linear(opt.doc_len, opt.id_emb_size, bias=False)
        self.item_projection = nn.Linear(opt.doc_len, opt.id_emb_size, bias=False)




    def forward(self, u_fea, i_fea):
        affinity = torch.matmul(u_fea, self.affinity_matrix)
        affinity = F.relu(torch.matmul(affinity, i_fea.T))

        u_proj = self.user_projection(u_fea)
        i_proj = self.item_projection(u_fea)

        u_aux = torch.matmul(affinity, u_proj)
        i_aux = torch.matmul(affinity.T, i_proj)

        u_fea = F.relu(u_proj + i_aux)
        i_fea = F.relu(i_proj + u_aux)

        u_fea = F.softmax(u_proj * u_fea, dim=1)
        i_fea = F.softmax(i_proj * i_fea, dim=1)

        return u_fea, i_fea


# class AspectImportanceEstimation(nn.Module):
#     def __init__(self, opt):
#         super(AspectImportanceEstimation, self).__init__()

#         self.opt = opt

#         self.affinity_matrix = nn.Parameter(torch.Tensor(10, 10), requires_grad=True)

#         # User "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
#         self.user_projection = nn.Parameter(torch.Tensor(32, 10), requires_grad = True)
#         self.user_parameters = nn.Parameter(torch.Tensor(32, 1), requires_grad = True)

# 		# Item "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
#         self.item_projection = nn.Parameter(torch.Tensor(32, 10), requires_grad = True)
#         self.item_parameters = nn.Parameter(torch.Tensor(32, 1), requires_grad = True)

#         # Reset Para
#         self.affinity_matrix.data.uniform_(-0.01, 0.01)

#         self.user_projection.data.uniform_(-0.01, 0.01)
#         self.user_parameters.data.uniform_(-0.01, 0.01)

#         self.item_projection.data.uniform_(-0.01, 0.01)
#         self.item_parameters.data.uniform_(-0.01, 0.01)


#     def forward(self, user_rep, item_rep):
#         user_rep_t = torch.transpose(user_rep, 1, 2)
#         item_rep_t = torch.transpose(item_rep, 1, 2)
        
#         # === USER IMPORTANCE ASPECTS === #

#         # Eq 5
#         affinity_matrix = torch.matmul(user_rep, self.affinity_matrix)
#         affinity_matrix = torch.matmul(affinity_matrix, item_rep_t)
#         affinity_matrix = F.relu(affinity_matrix)

#         # Eq 6
#         h_user = torch.matmul(self.user_projection, user_rep_t)
#         h_item = torch.matmul(self.item_projection, item_rep_t)
#         h_item = torch.matmul(h_item, torch.transpose(affinity_matrix, 1, 2))
#         h_user = F.relu(h_user + h_item)

#         user_asp_importance = torch.matmul(torch.transpose(self.user_parameters, 0, 1), h_user)
#         user_asp_importance = torch.transpose(user_asp_importance, 1, 2)
#         user_asp_importance = F.softmax(user_asp_importance, dim=1)
#         user_asp_importance = user_asp_importance.squeeze(2)

#         # === ITEM IMPORTANCE ASPECTS === #
        

#         # Eq 6
#         h_item = torch.matmul(self.item_projection, item_rep_t)
#         h_user = torch.matmul(self.user_projection, user_rep_t)
#         h_user = torch.matmul(h_user, affinity_matrix)
#         h_item = F.relu(h_item + h_user)

#         item_asp_importance = torch.matmul(torch.transpose(self.item_parameters, 0, 1), h_item)
#         item_asp_importance = torch.transpose(item_asp_importance, 1, 2)
#         item_asp_importance = F.softmax(item_asp_importance, dim=1)
#         item_asp_importance = item_asp_importance.squeeze(2)

#         return user_asp_importance, item_asp_importance


class RatingPrediction(nn.Module):
    def __init__(self, opt):
        super(RatingPrediction, self).__init__()

        self.opt = opt

        self.user_dropout = nn.Dropout(opt.drop_out)
        self.item_dropout = nn.Dropout(opt.drop_out)

        self.global_offset = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.global_offset.data.fill_(0)

        self.uid_offset = nn.Embedding(opt.user_num + 2, 1)
        self.uid_offset.weight.requires_grad = True
        self.uid_offset.weight.data.fill_(0)
        self.iid_offset = nn.Embedding(opt.item_num + 2, 1)
        self.iid_offset.weight.requires_grad = True
        self.iid_offset.weight.data.fill_(0)


    def forward(self, user_rep, item_rep, user_imp, item_imp, uids, iids):

        user_offset = self.uid_offset(uids)
        item_offset = self.iid_offset(iids)

        user_rep = self.user_dropout(user_rep)
        item_rep = self.item_dropout(item_rep)

        user_rep = torch.transpose(user_rep, 0, 1)
        item_rep = torch.transpose(item_rep, 0, 1)


        lstAspRating = []
        for k in range(5):
            user = user_rep[k].unsqueeze(1)
            item = item_rep[k].unsqueeze(2)

            asp_rating = torch.matmul(user, item)
            asp_rating = asp_rating.squeeze(2)
            
            lstAspRating.append(asp_rating)

        rating_pred = torch.cat(lstAspRating, dim=1)

        rating_pred = user_imp * item_imp * rating_pred
        rating_pred = torch.sum(rating_pred, dim=1, keepdim=True)
        rating_pred = rating_pred + user_offset + item_offset
        rating_pred = rating_pred + self.global_offset

        return rating_pred.squeeze(1)


# TESTING 
# class AspectRepresentationLearning(nn.Module):
#     def __init__(self, opt):
#         super(AspectRepresentationLearning, self).__init__()

#         self.opt = opt

#         self.aspects = nn.Parameter(torch.Tensor(5, opt.word_dim, 32), requires_grad=True)
#         self.conv = nn.Conv2d(opt.doc_len, opt.filters_num, (opt.kernel_size, 32), padding=1)
#         self.attention = nn.Sequential(
#             nn.Linear(opt.kernel_size * 5, 32),
#             nn.Softmax(dim=1)
#         )
#         # self.unfold = nn.Unfold((opt.kernel_size, 32), padding=(1, 0))


#     def forward(self, doc):
#         # Eq 1
#         aspect_projection = torch.matmul(doc.unsqueeze(1), self.aspects) # o_shape (bs, n_asp, doc_len, h1)
#         aspect_projection = self.conv(aspect_projection.transpose(1, 2))
#         aspect_projection = self.attention(aspect_projection.view(aspect_projection.shape[0],
#                                                                   aspect_projection.shape[1],
#                                                                   -1))
#         aspect_projection = torch.sum(aspect_projection, dim=1)
#         # aspect_projection = self.unfold(aspect_projection).transpose(1,2)
#         # window = aspect_projection.shape[2]


#         return aspect_projection

