# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataset import load_dataset
from net import *
from torch import linalg as LA
import logging
import time

import os.path

dataset='boston'
input_path='Data/'+dataset+'.npy'
label_path='Data/'+dataset+'-label.npy'

seed=0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

covariate, response, train_data, test_data=load_dataset(dataset, input_path, label_path)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

def l2_clipper(tensor,epsilon):
    norm=LA.vector_norm(tensor,ord=2,dim=1)
    tensor[norm>epsilon]=epsilon*tensor[norm>epsilon]/norm[norm>epsilon].unsqueeze(1)
    return tensor

def attack_fgsm(model, X, y, epsilon):
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
    delta=l2_clipper(delta, epsilon)
    delta.requires_grad = True
    output = model(X + delta)
    #loss = criterion(output.squeeze(1), y)
    loss = criterion(output,y.unsqueeze(1))
    
    loss.backward()
    grad = delta.grad.detach()
    delta.data = l2_clipper(delta + alpha * torch.sign(grad),epsilon)
    delta = delta.detach()
    return delta


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
    delta=l2_clipper(delta, epsilon)
    for _ in range(attack_iters):
        delta.requires_grad = True
        output = model(X + delta)
        #loss = criterion(output.squeeze(1), y)
        loss = criterion(output,y.unsqueeze(1))
        opt.zero_grad()
        loss.backward()
        grad = delta.grad.detach()
        delta.data = l2_clipper(delta + alpha * torch.sign(grad),epsilon)
        
    delta = delta.detach()
    return delta

def regressor_train():
    nepochs=400
    attack='pgd'
    lr_max=1e-3
    epsilon=0.2
    alpha=0.01
    attack_iters=50
    lr_type='flat'
    model=regressor(xdim=covariate.shape[1]).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    fname='models/'+dataset+'-'+attack+str(epsilon)+'.pth'

    if lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, nepochs * 2//5, nepochs], [0, lr_max, 0])[0]
    elif lr_type == 'flat': 
        lr_schedule = lambda t: lr_max
    else:
        raise ValueError('Unknown lr_type')

    criterion = nn.MSELoss()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG)
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Test Loss')
    for epoch in range(nepochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        
                    
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            X=X.float()
            y=y.float()
            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)

            if attack == 'fgsm':
                delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
                delta=l2_clipper(delta, epsilon)
                delta.requires_grad = True
                output = model(X + delta)
                #loss = criterion(output.squeeze(1), y)
                loss = criterion(output,y.unsqueeze(1))
                
                loss.backward()
                grad = delta.grad.detach()
                delta.data = l2_clipper(delta + alpha * torch.sign(grad),epsilon)
                delta = delta.detach()
            elif attack == 'none':
                delta = torch.zeros_like(X)
            elif attack == 'pgd':
                delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
                delta=l2_clipper(delta, epsilon)
                for _ in range(attack_iters):
                    delta.requires_grad = True
                    output = model(X + delta)
                    #loss = criterion(output.squeeze(1), y)
                    loss = criterion(output,y.unsqueeze(1))
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = l2_clipper(delta + alpha * torch.sign(grad),epsilon)
                    
                delta = delta.detach()
            
            output = model(X + delta)
            loss = criterion(output, y.unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_n += y.size(0)
        
        train_time = time.time()
        test_loss = 0
        test_acc = 0
        n = 0
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            X=X.float()
            #y=y.float()
            y=y.type(torch.cuda.LongTensor)
            
            with torch.no_grad():
                output=model(X)
                loss = criterion(output, y.unsqueeze(1))
                test_loss += loss.item() * y.size(0)
                n += y.size(0)
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, test_loss/n)
        torch.save(model.state_dict(), fname)
        
#regressor_train()


def Train(model,attack_net,opt,opt_attack,nepochs,headstart,alpha):
    print("***Adverserial training initiated****")
    logger.info('Epoch \t Time \t LR \t \t Train Loss ')
    LOSS=[]
    PGDLOSS=[]
    losspgdfgsm_buff=0
    losspgdpgd_buff=0
    for epoch in range(nepochs):
        start_time = time.time()
        for iters in range(headstart):
            #opt_attack.zero_grad()
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()
                X=X.float()
                y=y.float()
                
                delta=attack_net(X)
                output = model(X + delta)

                loss = -criterion(output, y.unsqueeze(1))

                opt_attack.zero_grad()
                opt.zero_grad()
                

                loss.backward() #for attack with lab use loss2
                opt_attack.step()
                
                
        
        for def_iters in range(defense_iters):
            train_loss = 0
            train_acc = 0
            train_n = 0        
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()
                X=X.float()
                y=y.float()
                lr = lr_schedule(epoch + (i+1)/len(train_loader))
                opt.param_groups[0].update(lr=lr)
                
                delta = attack_net(X)
                output = model(X + delta)
                loss = (1-alpha)*criterion(output, y.unsqueeze(1))+alpha*criterion(model(X),y.unsqueeze(1))
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * y.size(0)
                train_n += y.size(0)

        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr_attack, train_loss/train_n)
        
        loss_pgd,loss_fgsm,loss_nn=Test(model, test_loader,'nn',epoch)
        PGDloss_pgd,PGDloss_fgsm,PGDloss_nn=Test(pgd_model, test_loader,'pgd',epoch)
        if epoch ==0:
            losspgdfgsm_buff=PGDloss_fgsm
            losspgdpgd_buff=PGDloss_pgd
        LOSS.append(np.array((loss_pgd,loss_fgsm,loss_nn)))
        PGDLOSS.append(np.array((losspgdpgd_buff,losspgdfgsm_buff,PGDloss_nn)))
    print("***Training completed****")
    return LOSS,PGDLOSS

def Test(model,test_loader,DEF,epoch):
    test_loss = 0
    n = 0
    pgd_attack_loss=0
    fgsm_attack_loss=0
    nn_attack_loss=0
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        X=X.float()
        y=y.float()
        if DEF =='pgd' and epoch ==0:
            delta_pgd = attack_pgd(model, X, y, epsilon, 0.01, 50, 10)
            delta_fgsm = attack_fgsm(model, X, y, epsilon)
        elif DEF =='nn':
            delta_pgd = attack_pgd(model, X, y, epsilon, 0.01, 50, 10)
            delta_fgsm = attack_fgsm(model, X, y, epsilon)
        
        delta_nn = attack_net(X)
        
        with torch.no_grad():
            
            output = model(X)
            
            loss = criterion(output, y.unsqueeze(1)) 
            
            test_loss += loss.item() * y.size(0)
            if DEF =='pgd' and epoch ==0:
                pgd_loss= criterion(model(X+delta_pgd), y.unsqueeze(1)) # pgd attack loss
                fgsm_loss= criterion(model(X+delta_fgsm), y.unsqueeze(1)) # fgsm attack loss
                pgd_attack_loss += pgd_loss.item()
                fgsm_attack_loss += fgsm_loss.item()
                
            elif DEF =='nn':
                pgd_loss= criterion(model(X+delta_pgd), y.unsqueeze(1)) # pgd attack loss
                fgsm_loss= criterion(model(X+delta_fgsm), y.unsqueeze(1)) # fgsm attack loss
                pgd_attack_loss += pgd_loss.item()
                fgsm_attack_loss += fgsm_loss.item()
            
            nn_loss= criterion(model(X+delta_nn), y.unsqueeze(1)) # nn attack loss
            nn_attack_loss += nn_loss.item()
            n += y.size(0)
    logger.info('Test Loss : \t %.4f ', test_loss/n)
      
    return pgd_attack_loss,fgsm_attack_loss,nn_attack_loss


model=regressor(xdim=covariate.shape[1]).cuda()


""" attack network"""
epsilon=0.7
alpha=0.9
epsilon=round(epsilon,3)
attack_net=lp_regression_atk(epsilon=epsilon,p=2,xdim=covariate.shape[1]).cuda()



"""Adverserial training parameters"""
nepochs=400


""" Loading previously adv trained pgd model"""
pgd_model=regressor(xdim=covariate.shape[1]).cuda()
fname='models/'+dataset+'-'+'pgd'+str(epsilon)+'.pth'
pgd_model.load_state_dict(torch.load(fname))


alpha=round(alpha,2)
headstart=1  #how much the attck network must be trained before defense updates of the model
defense_iters=1 # no. of defense updates at once

lr_classifier=1e-3
lr_attack=2*1e-4#1e-3


lr_type='flat'
"""Training configuration"""

""" save path """
defense='NNadv-'
dpath= 'Moon Data diagnosis/paper_reg/'+dataset+'-'+defense+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
apath= 'Moon Data diagnosis/paper_reg/'+dataset+'-'+'advnn'+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'

"""optimisers for the classifier and attack model respectively"""
#opt = torch.optim.Adam(model.parameters(), lr=lr_classifier)
#opt_attack= torch.optim.Adam(attack_net.parameters(), lr=lr_attack)

opt = torch.optim.Adam(model.parameters(), lr=lr_classifier)
opt_attack= torch.optim.Adam(attack_net.parameters(), lr=lr_attack)        

#Scheduler
if lr_type == 'cyclic': 
    lr_schedule = lambda t: np.interp([t], [0, nepochs * 2//5, nepochs], [0, lr_classifier, 0])[0]
elif lr_type == 'flat': 
    lr_schedule = lambda t: lr_classifier
else:
    raise ValueError('Unknown lr_type')


criterion = nn.MSELoss()
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

LS,PGDLS=Train(model, attack_net, opt, opt_attack, nepochs, headstart,alpha)
loss_array=np.vstack(LS)
pgdloss_array=np.vstack(PGDLS)
file_name='Moon Data diagnosis/paper_reg/'+dataset+"-"+str(alpha)+"-alpha-epsilon-"+str(epsilon)

np.save(file_name+"LOSS.npy",loss_array)
np.save(file_name+"PGDLOSS.npy",pgdloss_array)
