# -*- encoding: utf-8 -*-
import time
import random
import math
import fire
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from dataset import ReviewData
from framework import Model
import models
import config
from utils.utils import *

from metrics.ndcg import ndcg_metric

# @TRAIN FUNCTION 
def train(**kwargs):

    # DATASET LOADING
    if 'dataset' not in kwargs:
        raise Exception("Dataset not provided.")
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()

    opt.parse(kwargs) 

    # PARALLEL CONFIG
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # IMPORT MODEL
    if 'model' not in kwargs:
        raise Exception("Model not provided.")
    model = Model(opt, getattr(models, opt.model))
    print(f"Model: {opt.model}")

    # SENDING MODEL TO GPU, IF EXISTS
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    # EXCEPTION FEATURES ERROR
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")
    
    # LOAD INTERACTIONS
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f'train data: {len(train_data)}; test data: {len(val_data)}')

    # OPTMIZERS
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # TRAINING STAGE
    print("start training....")
    min_loss = 1e+10
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()

    for epoch in range(opt.num_epochs):

        total_loss = 0.0
        total_maeloss = 0.0

        # MODEL IN TRAIN STAGE
        model.train()
        print(f"{now()}  Epoch {epoch}...")

        for idx, (train_datas, scores) in enumerate(train_data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
                
            # UNPACK NEDDED INPUTS
            train_datas = unpack_input(opt, train_datas)

            # RESET GRADIENT FOR BACKPROPAGATION
            optimizer.zero_grad()

            # PREDICT OUTPUT
            output = model(train_datas)

            # CALCULATE LOSSES
            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()
            smooth_mae_loss = smooth_mae_func(output, scores)

            # LOSS FUNCTION METHOD
            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss

            # BACKPROPAGATION
            loss.backward()
            optimizer.step()

            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    print("\t{}, {} step finised;".format(now(), idx))
                    val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
                    if val_loss < min_loss:
                        model.save(name=opt.dataset, opt=opt.emb_opt)
                        min_loss = val_loss
                        print("\tmodel save")
                    if val_loss > min_loss:
                        best_res = min_loss

        # SMOOTHING BACKPROPAGATION
        scheduler.step()
        
        mse = total_loss * 1.0 / len(train_data)
        print(f"\ttrain data: loss:{total_loss:.4f}, mse: {mse:.4f};")

        # VALIDATION TESTING FOR MODEL SAVE
        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
        if val_loss < min_loss:
            model.save(name=opt.dataset, opt=opt.emb_opt)
            min_loss = val_loss
            print("model save")
        if val_mse < best_res:
            best_res = val_mse
        print("*"*30)

    print("----"*20)
    print(f"{now()} {opt.dataset} {opt.emb_opt} best_res:  {best_res}")
    print("----"*20)


# @TEST FUNCTION
def test(**kwargs):

    if 'dataset' not in kwargs:
        raise Exception("Dataset not provided.")
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    opt.pth_path = f"checkpoints/{opt.model}_{opt.dataset}_{opt.emb_opt}.pth"

    # PARALLEL CONFIG
    assert(len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{now()}: test in the test dataset")
    predict_loss, test_mse, test_mae = predict(model, test_data_loader, opt)


# @PREDICT OUTPUT FUNCTION
def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0

    mse_values = []
    mae_values = []
    rmse_values = []
    ndcg_values = []
    precision_values = []
    recall_values = []
    
    # MODEL IN EVALUTATION STAGE
    model.eval()

    # WITHOUT GRADIENT
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)

            test_data = unpack_input(opt, test_data)

            output = model(test_data)

            mse_loss = torch.mean((output-scores)**2)
            total_loss += mse_loss.item()

            mae_loss = torch.mean(abs(output-scores))
            total_maeloss += mae_loss.item()

            rmse, precision, recall = calculate_metrics(scores, output)

            mse_values.append(mse_loss.cpu().item())
            rmse_values.append(rmse.item())
            mae_values.append(mae_loss.cpu().item())
            precision_values.append(precision.item())
            recall_values.append(recall.item())

    if opt.ranking_metrics:
        iteractions, scores = next(iter(data_loader))
        
        user_ids = set([x[0] for x in iteractions])
        item_ids = set([x[1] for x in iteractions])

        with torch.no_grad():
            for user in user_ids:
                test_data = unpack_input(opt, zip([user]*len(item_ids), item_ids))

                output = model(test_data)

                iids, output = test_data[3].cpu(), output.cpu()
                iids = [x.item() for x in iids]
                output = [x.item() for x in output]


                item_x_rating = list(zip(iids, output))
                item_x_rating.sort(key=lambda x: x[1])

                list_wise = [x[0] for x in item_x_rating]

                grownd_truth = [y[1] for y in [x for x in iteractions if user == x[0]]]

                # ndcg = ndcg_metric(grownd_truth, list_wise, nranks=4)

                # ndcg_values.append(ndcg)


    
    if opt.statistical_test:
        df = {
            "mse":mse_values, 
            "mae":mae_values,
            }
        
        df = pd.DataFrame(df)
        df.to_csv(f"results/{opt.model}_{opt.dataset}_{opt.emb_opt}_results.csv", index=False)

    else:
        print(f'''MSE mean: {np.array(mse_values).mean():.2f},
                MAE mean: {np.array(mae_values).mean():.2f}, 
                RMSE mean: {np.array(rmse_values).mean():.2f},  
                PRECISION mean: {np.array(precision_values).mean():.2f},
                RECALL mean: {np.array(recall_values).mean():.2f}'''
                
            )

            

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    
    # RETURN TO TRAIN STAGE
    model.train()

    return total_loss, mse, mae


# @UNPACK INTERACTIN FEATURES
def unpack_input(opt, x):

    uids, iids = list(zip(*x))
    uids = list(uids)
    iids = list(iids)
    
    user_reviews = opt.users_review_list[uids]
    user_item2id = opt.user2itemid_list[uids] 

    
    user_doc = opt.user_doc[uids]
    
    item_reviews = opt.items_review_list[iids]
    item_user2id = opt.item2userid_list[iids] 
    item_doc = opt.item_doc[iids]

    if opt.topics:
        user_doc = opt.topic_matrix[uids]

        shift = opt.user_num - 2
        item_doc = opt.topic_matrix[[x + shift for x in iids]] 
        
    
    data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
    data = list(map(lambda x: torch.LongTensor(x).cuda(), data))

    return data


if __name__ == "__main__":
    fire.Fire()