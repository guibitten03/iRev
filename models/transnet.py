# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random


class TRANSNET(nn.Module):
    
    def __init__(self, opt, uori='user'):
        super(TRANSNET, self).__init__()
        self.opt = opt
        self.num_fea = 1 # DOC
        
        self.source_net = SourceNet(opt)
        self.target_net = TargetNet(opt)
    

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user2item, item2user, _, _ = datas

        item_reviews = item_reviews[:, :user_reviews.shape[1], :]
        item2user = item2user[user2item.shape[1]]

        source_latent, source_prediction = self.source_net(
            user_reviews, item_reviews, uids, iids
        )

        # Get review of user i for item j in batch
        reviews = []
        for i in range(len(uids)):
            itens_of_user = user2item[i]
            item_of_batch = iids[i]

            if item_of_batch not in itens_of_user:
                users_of_item = item2user[i]
                user_of_batch = uids[i]

                if user_of_batch not in users_of_item:
                    target_review = torch.randint(0, self.opt.vocab_size, (self.opt.r_max_len, )).to('cuda')

                else:
                    idx_user = torch.where(users_of_item == user_of_batch)[0][0].item()
                    target_review = item_reviews[i][idx_user]             


            else:
                idx_item = torch.where(itens_of_user == item_of_batch)[0][0].item()
                target_review = user_reviews[i][idx_item]
                        
            reviews.append(target_review)

        reviews = torch.stack(reviews).to('cuda')

        target_latent, target_prediction = self.target_net(
            reviews
        )


        # return [source_latent, source_prediction.squeeze()], [target_latent, target_prediction.squeeze()]
        return source_latent, target_latent



class CNN(nn.Module):

    def __init__(self, opt, uori_len):
        super(CNN, self).__init__()

        self.opt = opt

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=opt.word_dim,
                out_channels=opt.filters_num,
                kernel_size=opt.kernel_size,
                padding=(opt.kernel_size - 1) // 2),  # out shape(new_batch_size, kernel_count, review_length)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, opt.r_max_len)),  # out shape(new_batch_size,kernel_count,1)
        )

        self.linear = nn.Sequential(
            nn.Linear(opt.filters_num, opt.id_emb_size),
            nn.Tanh(),
        )

    def forward(self, vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        latent = self.conv(vec.permute(0, 2, 1))  # output shape(new_batch_size, kernel_count, 1)
        latent = latent.view(-1, self.opt.filters_num * 1)
        latent = self.linear(latent)
        return latent  # output shape(batch_size, id_emb_size)
    

class SourceNet(nn.Module):

    def __init__(self, opt, extend_model=False):
        super(SourceNet, self).__init__()

        self.opt = opt

        self.extend_model = extend_model
        self.user_emb = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_emb = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.cnn_u = CNN(opt, opt.u_max_r)
        self.cnn_i = CNN(opt, opt.i_max_r)
        self.transform = nn.Sequential(
            nn.Linear(opt.id_emb_size * opt.u_max_r * 2, opt.id_emb_size),
            nn.Tanh(),
            nn.Linear(opt.id_emb_size, opt.id_emb_size),
            nn.Tanh(),
            nn.Dropout(opt.drop_out)
        )

        for m in self.transform.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0, std=0.1).clamp_(-1, 1)
                nn.init.constant_(m.bias.data, 0.1)

        
        self.fm = FactorizationMachine(in_dim=opt.id_emb_size, k=8)

    def forward(self, user_reviews, item_reviews, user_ids, item_ids):  # shape(batch_size, review_count, review_length)
        older_bs = user_reviews.shape[0]

        new_batch_size = user_reviews.shape[0] * user_reviews.shape[1]
        user_reviews = user_reviews.reshape(new_batch_size, -1)
        item_reviews = item_reviews.reshape(new_batch_size, -1)

        u_vec = self.user_emb(user_reviews)
        i_vec = self.item_emb(item_reviews)

        user_latent = self.cnn_u(u_vec)
        item_latent = self.cnn_i(i_vec)

        user_latent = user_latent.reshape(older_bs, self.opt.u_max_r, self.opt.id_emb_size)\
                        .reshape(older_bs, -1)
        item_latent = item_latent.reshape(older_bs, self.opt.u_max_r, self.opt.id_emb_size)\
                        .reshape(older_bs, -1)
        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        trans_latent = self.transform(concat_latent)

        prediction = self.fm(trans_latent.detach())  # Detach forward

        return trans_latent, prediction

    def trans_param(self):
        return [x for x in self.cnn_u.parameters()] + \
               [x for x in self.cnn_i.parameters()] + \
               [x for x in self.transform.parameters()]


class TargetNet(nn.Module):

    # Recebe uma review caso coincida o batch
    def __init__(self, opt):
        super(TargetNet, self).__init__()
        self.embedding = nn.Embedding(opt.vocab_size, opt.word_dim)

        self.conv = CNN(opt, opt.r_max_len)

        self.fm = nn.Sequential(
            nn.Dropout(opt.drop_out),  # Since cnn did not dropout, dropout before FM.
            FactorizationMachine(in_dim=opt.id_emb_size, k=8)
        )

    def forward(self, reviews):  # input shape(batch_size, review_length)
        vec = self.embedding(reviews)
        cnn_latent = self.conv(vec)
        prediction = self.fm(cnn_latent)
        return cnn_latent, prediction
    

class FactorizationMachine(nn.Module):

    def __init__(self, in_dim, k):  # in_dim=id_emb_size
        super(FactorizationMachine, self).__init__()
        self.v = nn.Parameter(torch.full([in_dim, k], 0.001))
        self.linear = nn.Linear(in_dim, 1)
        self.linear.weight.data.normal_(mean=0, std=0.001)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, id_emb_size), output shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 ** 2 - inter_part2, dim=1)
        output = linear_part.t() + 0.5 * pair_interactions
        return output.view(-1, 1)  # output shape(batch_size, 1)