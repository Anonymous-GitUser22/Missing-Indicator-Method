# Code is based on the official SAINT github repository
# https://github.com/somepago/saint

import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn as nn
import pickle

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler

def classification_scores(model, dloader, device, model_type='transformer', classes=2):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            cls, x, y_gts = data[0].to(device), data[1].float().to(device), data[2].long().to(device)

            if model_type=='mlp':
                y_outs = model(x)
            else:
                n1,n2 = x.shape
                x_embedded = torch.empty(n1, n2, model.n_embed)
                for i in range(n2):
                    x_embedded[:,i,:] = model.embeds[i](x[:,i])
                x_embedded = x_embedded.to(device)
                # Adding embedded CLS Token to the start (convention to add to start?)
                cls_embedded = model.cat_embeds(cls)
                x_embedded = torch.cat((cls_embedded,x_embedded),dim=1)
                
                reps = model.transformer(x_embedded)
                # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
                y_reps = reps[:,0,:]
                y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    if classes==2:
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    else: 
        auc = 0
    return acc.cpu().numpy(), auc

def save_dataset_dict(dict, path, name):
    with open(f'{path}/{name}_dictionary.pickle', 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

def save_dataset(dset, path, name):
    with open(f'{path}/{name}_dataset.pickle', 'wb') as f:
        pickle.dump(dset, f, pickle.HIGHEST_PROTOCOL)