# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MAN(nn.Module):
   
    def __init__(self, opt, uori='user'):
        super(MAN, self).__init__()
        self.opt = opt
        self.num_fea = 1 # DOC
        

        self.user_review_emb = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_review_emb = nn.Embedding(opt.vocab_size, opt.word_dim)

        # Convolution Kernal shape: (T, word_dim) - padding=same
        # Slices - [2,3,4]
        # Convolution is done by reviews of user to various itens
        self.user_conv = TemporalConvolutionNetwork(opt)
        self.item_conv = TemporalConvolutionNetwork(opt, uori='item')

        self.ifl = InteractionFeatureLearning(opt)

        self.uid_embedding = nn.Embedding(opt.user_num + 2, opt.id_emb_size)
        self.iid_embedding = nn.Embedding(opt.item_num + 2, opt.id_emb_size)

        self.auxiliar = AuxiliaryNetwork(opt)


        # self.reset_para()   

    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas


        user_reviews = self.user_review_emb(user_doc)
        item_reviews = self.item_review_emb(item_doc)

        pad = (3 - 1) // 2
        user_tcn = F.pad(user_reviews, (0, 0, pad, pad), 'constant', 0)
        item_tcn = F.pad(item_reviews, (0, 0, pad, pad), 'constant', 0)

        # TemporalConvolutionNetwork
        # user_tcn = self.user_conv(user_tcn.unsqueeze(1))
        # item_tcn = self.item_conv(item_tcn.unsqueeze(1))
        user_tcn = self.user_conv(user_reviews.unsqueeze(1))
        item_tcn = self.item_conv(item_reviews.unsqueeze(1))

        euclidean = (user_tcn - item_tcn.permute(0, 1, 3, 2)).pow(2).sum(1).sqrt()
        attention_matrix = 1.0 / (1 + euclidean)
        attention_matrix = torch.softmax(torch.sum(attention_matrix, 2), dim=1)
        u_review_level = user_reviews * attention_matrix.unsqueeze(2)
        i_review_level = item_reviews * attention_matrix.unsqueeze(2)

        pad = (3 - 1) // 2
        u_review_level = F.pad(u_review_level, (0, 0, pad, pad), 'constant', 0)
        i_review_level = F.pad(i_review_level, (0, 0, pad, pad), 'constant', 0)

        ui_features = self.ifl(u_review_level.unsqueeze(1), i_review_level.unsqueeze(1)) # (128, 300)

        u_emb = self.uid_embedding(uids)
        i_emb = self.iid_embedding(iids)
        pred_hidden_fea = torch.cat([ui_features, u_emb, i_emb], dim=1)

        # ui_fea = torch.cat([user_doc, item_doc], dim=1) # (128, 1000)
        # ui_fea = self.auxiliar(ui_fea)

        return pred_hidden_fea, None

    def reset_para(self):

        for cnn in [self.user_cnn, self.item_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for fc in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

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


class TemporalConvolutionNetwork(nn.Module):

    def __init__(self, opt, uori='user'):
        super(TemporalConvolutionNetwork, self).__init__()

        self.opt = opt

        '''
            Eq 2, 3, 4
        '''

        # kernels = [2,3,4]

        self.conv1 = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        # self.conv2 = nn.Conv2d(opt.filters_num, opt.filters_num, (opt.kernel_size, opt.word_dim))
        # if uori == 'user':
        #     self.conv3 = nn.Conv2d(opt.filters_num, opt.filters_num, (opt.kernel_size, opt.word_dim))
        # else:
        #     self.conv3 = nn.Conv2d(opt.filters_num, 1, (opt.kernel_size, opt.word_dim))


    def forward(self, x):

        fea = self.conv1(x)
        # fea = self.conv2(fea)
        # fea = self.conv3(fea)

        return fea


class InteractionFeatureLearning(nn.Module):
    def __init__(self, opt):
        super(InteractionFeatureLearning, self).__init__()

        self.opt = opt

        self.user_tcn = nn.Conv2d(1, 1, (opt.kernel_size, opt.word_dim))
        self.item_tcn = nn.Conv2d(1, 1, (opt.kernel_size, opt.word_dim))

        self.user_mlp = nn.Sequential(
            nn.Linear(opt.doc_len, opt.word_dim),
            nn.ReLU(),  
            nn.Linear(opt.word_dim, opt.word_dim),
            nn.ReLU(),
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(opt.doc_len, opt.word_dim),
            nn.ReLU(),
            nn.Linear(opt.word_dim, opt.word_dim),
            nn.ReLU(),
        )

        self.ui_features_mlp = nn.Sequential(
            nn.Linear(opt.word_dim * 2, opt.word_dim),
            nn.ReLU(),
            nn.Linear(opt.word_dim, opt.fc_dim),
            nn.ReLU()
        )



    def forward(self, user_fea, item_fea):
        
        user_fea = self.user_tcn(user_fea).squeeze(1)
        item_fea = self.item_tcn(item_fea).squeeze(1)

        user_fea = self.user_mlp(user_fea.permute(0, 2, 1))
        item_fea = self.item_mlp(item_fea.permute(0, 2, 1)) # (128, 300, 300)

        ui_fea = torch.cat([user_fea, item_fea], dim=1)
        ui_fea = ui_fea.view(ui_fea.shape[0], ui_fea.shape[1] * ui_fea.shape[2])
        
        ui_fea = self.ui_features_mlp(ui_fea)

        return ui_fea


class AuxiliaryNetwork(nn.Module):
    def __init__(self, opt):
        super(AuxiliaryNetwork, self).__init__()

        self.opt = opt

        self.ui_doc_emb = nn.Embedding(opt.doc_len * 2, opt.word_dim)

        self.word_att = nn.Sequential(
            nn.Linear(opt.word_dim, opt.word_dim),
            nn.ReLU(),
        )

        self.h = torch.randn(opt.word_dim, requires_grad=True)
        self.hb = torch.randn(opt.word_dim, requires_grad=True)

    def forward(self, ui_emb):

        ## Word-based Attention Layer
        ui_fea = self.ui_doc_emb(ui_emb) # (128, 1000, 300)
        attention_w = self.word_att(ui_fea) # (128, 1000, 300)

        attention_w = (self.h.T * attention_w) + self.hb
        attention_w = F.softmax(attention_w, dim=1)

        ui_fea = ui_fea * attention_w

        return ui_fea
