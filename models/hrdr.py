# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class HRDR(nn.Module):
    def __init__(self, opt):
        super(HRDR, self).__init__()
        self.opt = opt
        self.num_fea = 2  # ID + Review

        self.user_net = Net(opt, uori='user')
        self.item_net = Net(opt, uori='item')

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, _, _, _, _ = datas
        u_fea = self.user_net(user_reviews, uids, 'user')
        i_fea = self.item_net(item_reviews, iids, 'item')
        return u_fea, i_fea


class Net(nn.Module):
    def __init__(self, opt, uori='user'):
        super(Net, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num
        else:
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num

        '''
            Construct a matrix user_num x item_num of ids embeddings
        '''
        self.id_embedding = nn.Embedding(ui_id_num, opt.id_emb_size)
        self.rating_matrix = nn.Embedding(opt.user_num, opt.item_num)  # user/item num * 32
        self.rating_matrix.weight.requires_grad = False
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300

        '''
            Eq 1
        '''
        self.rating_mlp = nn.Sequential(
            nn.Linear(id_num, id_num // 2),
            nn.Tanh(),
            nn.Linear(id_num // 2, id_num // 4),
            nn.Tanh(),
            nn.Linear(id_num // 4, self.opt.filters_num),
            nn.LayerNorm(self.opt.filters_num)
        )

        self.cnn = nn.Conv2d(1, self.opt.filters_num, (self.opt.kernel_size, self.opt.word_dim))           

        self.id_linear = nn.Linear(self.opt.filters_num, self.opt.filters_num)
        self.review_linear = nn.Linear(self.opt.filters_num, self.opt.id_emb_size)


        self.dropout = nn.Dropout(self.opt.drop_out)

        self.init_word_emb()
        self.init_model_weight()

    def forward(self, reviews, ids, uori):
        
        id_embs = self.id_embedding(ids)

        # --------------- word embedding ----------------------------------
        
        reviews = self.word_embs(reviews) 
        reviews = self.dropout(reviews)
        bs, r_num, r_len, wd = reviews.size()
        reviews = reviews.view(-1, r_len, wd)
        
        '''
            Get respective id embedding: No grad.
        '''
        if uori == 'user':
            matrix_vector = self.rating_matrix(ids)
        else:
            matrix_vector = (self.rating_matrix.weight[:, ids]).t()


        '''
            Eq: 1
        '''
        Matrix_vector = self.rating_mlp(matrix_vector)
        
        '''
            Eq: 2, 3, 4
        '''
        review_fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3) 
        review_fea = F.max_pool1d(review_fea, review_fea.size(2)).squeeze(2)
        review_fea = review_fea.view(-1, r_num, review_fea.size(1)) # o = (128, 7, 100)

        attention_query = self.id_linear(Matrix_vector) # qr


        att_weight = F.softmax(torch.bmm(review_fea, attention_query.unsqueeze(2)), dim=1)

        weight_reviews = torch.sum(att_weight * review_fea, 1)

        review_fea = self.review_linear(weight_reviews)

        return torch.stack([id_embs, review_fea], 1)

    def init_word_emb(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)

        # ----------------Matrix init method--------------#
        ratingMatrix = torch.from_numpy(np.load(self.opt.ratingMatrix_path))
        self.rating_matrix.weight.data.copy_(ratingMatrix.cuda())
        # self.iid_embedding.weight.data.copy_(ratingMatrix.T.cuda())


    def init_model_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.uniform_(self.cnn.bias, a=-0.1, b=0.1)


        for x in [self.rating_mlp[0],
                  self.rating_mlp[2],
                  self.rating_mlp[4],
                  self.id_linear,
                  self.review_linear]:
            # nn.init.uniform_(x.weight, -0.1, 0.1)
            nn.init.xavier_normal_(x.weight)
            nn.init.constant_(x.bias, 0.1)



class CNN(nn.Module):
    '''
    for review and summary encoder
    '''

    def __init__(self, filters_num, k1, k2, padding=True):
        super(CNN, self).__init__()

        if padding:
            self.cnn = nn.Conv2d(1, filters_num, (k1, k2), padding=(int(k1 / 2), 0))
        else:
            self.cnn = nn.Conv2d(1, filters_num, (k1, k2))

    def multi_attention_pooling(self, x, qv):
        '''
        x: 704 * 100 * 224
        qv: 5 * 100
        '''
        att_weight = torch.matmul(x.permute(0, 2, 1), qv.t())  # 704 * 224 * 5
        att_score = F.softmax(att_weight, dim=1) * np.sqrt(att_weight.size(1))  # 704 * 224 *5
        x = torch.bmm(x, att_score)  # 704 * 100 * 5
        x = x.view(-1, x.size(1) * x.size(2))  # 704 * 500
        return x

    def attention_pooling(self, x, qv):
        '''
        x: 704 * 224 * 100
        qv: 704 * 100
        '''
        att_weight = torch.matmul(x, qv)
        att_score = F.softmax(att_weight, dim=1)
        x = x * att_score

        return x.sum(1)

    def forward(self, x, max_num, review_len, pooling="MAX", qv=None):
        '''
        eg. user
        x: (32, 11, 224, 300)
        multi_qv: 5 * 100
        qv: 32, 11, 100
        '''
        x = x.view(-1, review_len, self.cnn.kernel_size[1])
        x = x.unsqueeze(1)
        x = F.relu(self.cnn(x)).squeeze(3)
        if pooling == 'multi_att':
            assert qv is not None
            x = self.multi_attention_pooling(x, qv)
            x = x.view(-1, max_num, self.cnn.out_channels * qv.size(0))
        elif pooling == "att":
            x = x.permute(0, 2, 1)
            qv = qv.t()
            x = self.attention_pooling(x, qv)
            x = x.view(-1, max_num, self.cnn.out_channels)
        else:
            x = F.max_pool1d(x, x.size(2)).squeeze(2)  # B, F
            x = x.view(-1, max_num, self.cnn.out_channels)

        return x