# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import time

from .prediction import PredictionLayer
from .fusion import FusionLayer

class Model(nn.Module):
    # BASE NETWORK MODULE

    def __init__(self, opt, Net):
        # IMPORT OPTIONS AND NETWORK
        super(Model, self).__init__()

        self.opt = opt
        
        self.model_name = self.opt.model
        self.net = Net(opt)

        # DEFINE MERGE AND OUTPUT FEATURES 
        if self.opt.ui_merge == 'cat':
            if self.opt.r_id_merge == 'cat':
                    feature_dim = self.opt.id_emb_size * self.opt.num_fea * 2
            else:
                    feature_dim = self.opt.id_emb_size * 2
        else:
            if self.opt.r_id_merge == 'cat':
                    feature_dim = self.opt.id_emb_size * self.opt.num_fea
            else:
                    feature_dim = self.opt.id_emb_size


        if opt.man:
            feature_dim = self.opt.fc_dim + (self.opt.id_emb_size * 2)


        self.opt.feature_dim = feature_dim
        self.fusion_net = FusionLayer(opt)
        self.predict_net = PredictionLayer(opt)
        self.dropout = nn.Dropout(self.opt.drop_out)

    # @INPUT FORWARD THROW NETWORK
    def forward(self, datas):

        # UNPACK IMPUT
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        # THROW NETWORK
        user_feature, item_feature = self.net(datas)


        if self.opt.direct_output:
            return user_feature
        
        
        if self.opt.man: 
            ui_feature = self.dropout(user_feature)
            output = self.predict_net(ui_feature, uids, iids).squeeze(1)
            return output
        
        if self.opt.transnet:
            return user_feature, item_feature


        # THROW FUSION NET
        ui_feature = self.fusion_net(user_feature, item_feature)
        ui_feature = self.dropout(ui_feature)

        # THROW PREDICTION NET
        output = self.predict_net(ui_feature, uids, iids).squeeze(1)

        return output
    
    # @LOAD TRAINED NETWORKS
    def load(self, path):
        
        self.load_state_dict(torch.load(path))

    # @SAVE TRAINED NETWORKS
    def save(self, epoch=None, name=None, opt=None):
        
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + str(name) + '_' + str(opt) + '.pth'

        torch.save(self.state_dict(), name)
        
        return name