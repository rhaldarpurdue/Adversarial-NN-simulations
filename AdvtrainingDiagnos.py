# -*- coding: utf-8 -*-


import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

import logging
import time

import os.path

from net import *

from DecisionBoundary import plot_decision_boundary, plot_decision_boundary2D
from utils import clamp, attack_fgsm, attack_pgd

from sklearn import datasets
from dataset import load_dataset
from torch import linalg as LA
import probscale



def plot_attacks(pname,dataset,attack,test_loader,resolution):
    alpha=0.01
    attack_iters=50
    restarts=10
    xl, yl = covariate.min(axis=0) - 0.1
    xu, yu = covariate.max(axis=0) + 0.1
    
    # Successful attacks
    original_X=[]
    Perturbed_X=[]
    Labels=[]
    prob=[]
    prob_before=[]

    #All attacks
    original_X_all=[]
    Perturbed_X_all=[]
    Labels_all=[]
    prob_all=[]
    prob_before_all=[]
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        
        X=X.float()
        #y=y.float()
        y=y.type(torch.cuda.LongTensor)
        if attack == 'pgd':
            delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,trim)
        elif attack == 'fgsm':
            delta = attack_fgsm(model, X, y, epsilon,trim)
        elif attack == 'nnattack':
            if lab:
                delta = attack_net(X,y)
            else:
                delta = attack_net(X)
        elif attack == 'netG':
            delta= torch.clamp(attack_net(X),-epsilon,epsilon)
        
        with torch.no_grad():
            if trim:
                output = model(torch.clamp(X + delta,0,1))
            else:
                output = model(X + delta)
            
            index = torch.where(output.max(1)[1] != y)[0]
            shift=X+delta
            o=model(X)
            if len(index) != 0:
                
                prob_before.append(torch.nn.functional.softmax(o[index],dim=1).detach().to('cpu').numpy())
                prob.append(torch.nn.functional.softmax(output[index],dim=1).detach().to('cpu').numpy())
                Perturbed_X.append(shift[index].detach().to('cpu').numpy())
                Labels.append(y[index].detach().to('cpu').numpy())
                original_X.append(X[index].detach().to('cpu').numpy())
            
            
            prob_before_all.append(torch.nn.functional.softmax(o,dim=1).detach().to('cpu').numpy())
            prob_all.append(torch.nn.functional.softmax(output,dim=1).detach().to('cpu').numpy())
            Perturbed_X_all.append(shift.detach().to('cpu').numpy())
            Labels_all.append(y.detach().to('cpu').numpy())
            original_X_all.append(X.detach().to('cpu').numpy())
            
           
    if len(original_X)==0:
        print("no successful attacks")
    else:
        x=np.concatenate(original_X, axis=0 )
        y=np.concatenate(Labels, axis=0 )
        x2=np.concatenate(Perturbed_X, axis=0 )
        probability_before=np.concatenate(prob_before, axis=0)
        probability=np.concatenate(prob, axis=0 )
        k=np.concatenate((probability_before,probability),axis=1)
        
    

    x_all=np.concatenate(original_X_all, axis=0 )
    y_all=np.concatenate(Labels_all, axis=0 )
    x2_all=np.concatenate(Perturbed_X_all, axis=0 )
    probability_before_all=np.concatenate(prob_before_all, axis=0)
    probability_all=np.concatenate(prob_all, axis=0 )
    k_all=np.concatenate((probability_before_all,probability_all),axis=1)

    col=('red','blue','green')
    

    if len(original_X)==0:
        print("no plot")
    else:
        for i in range(len(x)):
            plt.xlim(xl, xu)
            plt.ylim(yl, yu)
            # plotting the corresponding x with y 
            # and respective color
            plt.quiver(x[i,0], x[i,1],x2[i,0]-x[i,0],x2[i,1]-x[i,1], color= col[y[i]],angles='xy',units='xy',scale_units='xy',scale=1,width=0.005)
           
              
          
        #plt.show()
        plt.savefig(pname+'.png',dpi=resolution)
        plt.close()

   
    for i in range(len(x_all)):
        plt.xlim(xl, xu)
        plt.ylim(yl, yu)
        # plotting the corresponding x with y 
        # and respective color
        plt.quiver(x_all[i,0], x_all[i,1],x2_all[i,0]-x_all[i,0],x2_all[i,1]-x_all[i,1], color= col[y_all[i]],angles='xy',units='xy',scale_units='xy',scale=1,width=0.005)
          
      
    #plt.show()
    plt.savefig(pname+'-all.png',dpi=resolution)
    plt.close()

def Pretrain(model,opt,pretrain_epochs):
    print("***Pretrain initiated****")
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc \t Test Acc')
    for epoch in range(pretrain_epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        
                       
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            X=X.float()
            #y=y.float()
            y=y.type(torch.cuda.LongTensor)
            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)
            
            output=model(X)
            
            loss = criterion(output.squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            #out=(output.squeeze(1)>0.5).float()
            #train_acc += (out== y).sum().item()
            train_acc += (output.max(1)[1] == y).sum().item()
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
                if trim:
                    output = model(torch.clamp(X,0,1))
                else:
                    output = model(X)
                loss = criterion(output.squeeze(1), y)
                test_loss += loss.item() * y.size(0)
                #out=(output.squeeze(1)>0.5).float()
                #test_acc += (out == y).sum().item()
                #print(torch.cat([output.max(1)[1].unsqueeze(1),y.unsqueeze(1)],dim=1))
                test_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
                
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n, test_acc/n)
    print("******Pretrain complete*****")
  
def BurnIn(model,attack_net,opt,opt_attack,burnin_epochs):
    print("***Burn in initiated****")
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc \t Fooled')
    for epoch in range(burnin_epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        
        X_all,grad_norm,grads,threshold=grad_dist(model, train_loader, percentile=0.9)
        
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            X=X.float()
            #y=y.float()
            y=y.type(torch.cuda.LongTensor)
            
            if lab:
                delta=attack_net(X,y)
                
            else:
                delta=attack_net(X)
            if signed_train:
                delta_clone=delta.detach()
                delta_clone.requires_grad=True
                output = model(X + delta_clone)
                output2 = model(torch.clamp(X + delta_clone, 0, 1))
            else:
                output = model(X + delta)
                output2 = model(torch.clamp(X + delta, 0, 1))
            
            
            index = torch.where(output.max(1)[1] == y)[0] #the samples which have not had a succesful attack 
            if len(index) == 0:
                break
            loss2 = -criterion(output2, y)
            loss = -criterion(output[index], y[index])
            opt_attack.zero_grad()
            opt.zero_grad()
            if signed_train:
                loss.backward()
                x_grad=delta_clone.grad.detach()
                gnorm=LA.vector_norm(x_grad,dim=1)
                gmax=torch.max(gnorm)
                x_grad[gnorm>threshold]=gmax*x_grad[gnorm>threshold]/gnorm[gnorm>threshold].unsqueeze(1)
                attack_net.zero_grad() # set parameters grad to 0
                delta.backward(x_grad)
                """parameter updates"""
                for param in attack_net.parameters():
                    param.data= param.data - lr_attack*param.grad
                
            else:
                loss.backward() #for attack with lab use loss2
                opt_attack.step()
            if trim:
                train_loss += -loss2.item() * y.size(0)
                train_acc += (output2.max(1)[1] == y).sum().item()
            else:
                train_loss += -loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
            
            train_n += y.size(0)

        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %d',
            epoch, train_time - start_time, lr_attack, train_loss/train_n, train_acc/train_n, train_n-train_acc)
   
    print("****Burn-in completed****")    
def grad_dist(model,train_loader,percentile):
    
        X_all=[]
        grads=[]
        #epsilon=0.3
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            X=X.float()
            #y=y.float()
            y=y.type(torch.cuda.LongTensor)
            X_clone=torch.clone(X)
            X_clone.requires_grad=True
            loss=sum_criterion(model(X_clone),y)
            loss.backward()
            grads.append(X_clone.grad.cpu().detach().numpy())
            X_clone.grad.zero_()
            X_all.append(torch.cat([X,y.unsqueeze(1)],dim=1))

        X_all=torch.cat(X_all,dim=0)
        X_all=X_all.cpu().detach().numpy()
        grads=np.vstack(grads)
        grad_norm=np.apply_along_axis(np.linalg.norm, 1, grads)
        threshold=np.percentile(grad_norm,q=(percentile)) #seems like top 2 % are the only relevant gradients
        ids=grad_norm>threshold
        gmax=np.max(grad_norm)
        sampled_data=torch.utils.data.TensorDataset(torch.from_numpy(X_all[ids,:][:,[0,1]]),torch.from_numpy(X_all[ids,2]))
        loader=torch.utils.data.DataLoader(sampled_data, batch_size=100, shuffle=False)
        return loader,X_all,grad_norm,grads,threshold, gmax
        ###############################################

def grad_plot(X_all,grads,threshold,file):
    grad_norm=np.apply_along_axis(np.linalg.norm, 1, grads)
    norm = mpl.colors.Normalize(vmin=np.min(grad_norm), vmax=np.max(grad_norm))
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    #Use scale =1 for same units as x,y coordinate
    plt.quiver(X_all[:,0], X_all[:,1],grads[:,0],grads[:,1],grad_norm,angles='xy',units='xy',scale_units='xy',cmap=cmap,scale=1)
    plt.colorbar(m)  
    plt.savefig(file+'grad_x.png',dpi=resolution)
    plt.close()
          
    probscale.probplot(grad_norm, plottype='pp',problabel='Percentiles',datascale='log')
    plt.savefig(file+'-pp.png',dpi=resolution)
    plt.close()
    
    idx=grad_norm>threshold
    max_norm=np.max(grad_norm)
    grads=grads/max_norm
    grads[idx,:]=max_norm*grads[idx,:]/grad_norm[idx,np.newaxis]  
    grad_norm=np.apply_along_axis(np.linalg.norm, 1, grads)
    
    norm = mpl.colors.Normalize(vmin=np.min(grad_norm), vmax=np.max(grad_norm))
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.quiver(X_all[:,0], X_all[:,1],grads[:,0],grads[:,1],grad_norm,angles='xy',units='xy',scale_units='xy',cmap=cmap,scale=1)
    plt.colorbar(m)  
    plt.savefig(file+'grad_x_normaise.png',dpi=resolution)
    plt.close()


def Train(model,attack_net,opt,opt_attack,nepochs,headstart,alpha):
    print("***Adverserial training initiated****")
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc \t Fooled')
    Acc=[]
    fools=[]
    PGDfools=[]
    LOSS=[]
    PGDLOSS=[]
    pgdfgsm_buff=0
    pgdpgd_buff=0
    losspgdfgsm_buff=0
    losspgdpgd_buff=0
    for epoch in range(nepochs):
        start_time = time.time()
        percentile=80
        loader,X_all,grad_norm,grads,threshold,gmax=grad_dist(model, train_loader, percentile)
        file='Moon Data diagnosis/Plots/grad/'+dataset+'-'+'eps-'+str(epsilon)+'alpha-'+str(alpha)+'-'+str(epoch)+'-'+ind
        """if (epoch%10)==0:
            grad_plot(X_all,grads,threshold,file)"""
        
        for iters in range(headstart):
            #opt_attack.zero_grad()
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()
                X=X.float()
                #y=y.float()
                y=y.type(torch.cuda.LongTensor)
                
                if lab:
                    delta=attack_net(X,y)
                    
                else:
                    delta=attack_net(X)
                if signed_train:
                    delta_clone=delta.detach()
                    delta_clone.requires_grad=True
                    output = model(X + delta_clone)
                    output2 = model(torch.clamp(X + delta_clone, 0, 1))
                else:
                    output = model(X + delta)
                    output2 = model(torch.clamp(X + delta, 0, 1))
                
                
                index = torch.where(output.max(1)[1] == y)[0] #the samples which have not had a succesful attack 
                if len(index) == 0:
                    break
                if CW_loss:
                    labels=y
                    loss=cw_loss(output, labels, N_cls, adv_lambda)
                    loss2=cw_loss(output2, labels, N_cls, adv_lambda)
                else:
                    loss2 = -sum_criterion(output2, y)
                    loss = -sum_criterion(output[index], y[index])
                    #loss = -sum_criterion(output, y) # for the loss plot
                
                #loss = -sum_criterion(output, y)# no index
                #print('iteration-',iters,'batch-',i,'loss:-',loss)
                
                opt_attack.zero_grad()
                opt.zero_grad()
                if signed_train:
                    loss.backward()
                    x_grad=delta_clone.grad.detach()
                    #grad_plot(X.cpu().detach().numpy(), -x_grad.cpu().detach().numpy(), threshold,file+str(i)+'batch')
                    gnorm=LA.vector_norm(x_grad,dim=1)
                    x_grad=x_grad/gmax
                    x_grad[gnorm>threshold]=gmax*x_grad[gnorm>threshold]/gnorm[gnorm>threshold].unsqueeze(1)
                    attack_net.zero_grad() # set parameters grad to 0
                    delta.backward(x_grad)
                    
                    """parameter updates
                    for param in attack_net.parameters():
                        param.data= param.data - lr_attack*param.grad
                        """
                    opt_attack.step()
                    """
                    #Plotting with each batch update
                    data_img='Moon Data diagnosis/Plots/AdvTrain/'+dataset+'-'+'eps-'+str(epsilon)+'alpha-'+str(alpha)+'-'+'iteration-'+str(iters)+'batch-'+str(i)+'-epochsDB.png'
                    data_img2='Moon Data diagnosis/Plots/AdvTrain/'+dataset+'-'+'eps-'+str(epsilon)+'alpha-'+str(alpha)+'-'+'iteration-'+str(iters)+'batch-'+str(i)+'-epochsATT.png'
                    resolution=600
                    for a,b in torch.utils.data.DataLoader(train_data,batch_size=len(train_data),shuffle=True):
                        a, b = a.cuda(), b.cuda()
                        a=a.float()
                        #y=y.float()
                        b=b.type(torch.cuda.LongTensor)
                        delta = attack_net(a,b) if lab else attack_net(a)
                        pert=a+delta
                        break
                    plot_decision_boundary(lambda x: model(x), pert.cpu().detach().numpy(), b.cpu().detach().numpy())
                    plt.savefig(data_img,dpi=resolution)
                    plt.close()
                    plot_attacks(data_img2, dataset, 'nnattack', test_loader, resolution)
                    """
                else:
                    loss.backward() #for attack with lab use loss2
                    opt_attack.step()
                    """
                    data_img='Moon Data diagnosis/Plots/AdvTrain/'+dataset+'-'+'eps-'+str(epsilon)+'alpha-'+str(alpha)+'-'+'iteration-'+str(iters)+'batch-'+str(i)+'-epochsDB.png'
                    data_img2='Moon Data diagnosis/Plots/AdvTrain/'+dataset+'-'+'eps-'+str(epsilon)+'alpha-'+str(alpha)+'-'+'iteration-'+str(iters)+'batch-'+str(i)+'-epochsATT.png'
                    resolution=600
                    for a,b in torch.utils.data.DataLoader(train_data,batch_size=len(train_data),shuffle=True):
                        a, b = a.cuda(), b.cuda()
                        a=a.float()
                        #y=y.float()
                        b=b.type(torch.cuda.LongTensor)
                        delta = attack_net(a,b) if lab else attack_net(a)
                        pert=a+delta
                        break
                    plot_decision_boundary(lambda x: model(x), pert.cpu().detach().numpy(), b.cpu().detach().numpy())
                    plt.savefig(data_img,dpi=resolution)
                    plt.close()
                    plot_attacks(data_img2, dataset, 'nnattack', test_loader, resolution)
                    """
            """
            if signed_train: #Doing whole batch update
                opt_attack.step()"""
                
        
        for def_iters in range(defense_iters):
            train_loss = 0
            train_acc = 0
            train_n = 0        
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()
                X=X.float()
                #y=y.float()
                y=y.type(torch.cuda.LongTensor)
                lr = lr_schedule(epoch + (i+1)/len(train_loader))
                opt.param_groups[0].update(lr=lr)
                
                delta = attack_net(X,y) if lab else attack_net(X)
                output = model(X + delta)
                output2 = model(torch.clamp(X + delta, 0, 1))
                
                loss = (1-alpha)*criterion(output, y)+alpha*criterion(model(X),y)
                loss2 = (1-alpha)*criterion(output2, y)+ alpha*criterion(model(X),y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if trim:
                    train_loss += loss2.item() * y.size(0)
                    train_acc += (output2.max(1)[1] == y).sum().item()
                else:
                    train_loss += loss.item() * y.size(0)
                    train_acc += (output.max(1)[1] == y).sum().item()
                
                train_n += y.size(0)
        

        train_time = time.time()
        if (epoch%90)==0:
            data_img='Moon Data diagnosis/Plots/AdvTrain/'+'L'+str(P)+' '+dataset+'-'+'eps-'+str(epsilon)+'alpha-'+str(alpha)+'-'+str(epoch)+'-'+ind+'-epochsDB.png'
            data_img2='Moon Data diagnosis/Plots/AdvTrain/'+'L'+str(P)+' '+dataset+'-'+'eps-'+str(epsilon)+'alpha-'+str(alpha)+'-'+str(epoch)+'-'+ind+'-epochsATT.png'
            
            for a,b in torch.utils.data.DataLoader(train_data,batch_size=len(train_data),shuffle=True):
                a, b = a.cuda(), b.cuda()
                a=a.float()
                #y=y.float()
                b=b.type(torch.cuda.LongTensor)
                delta = attack_net(a,b) if lab else attack_net(a)
                pert=a+delta
                break
            if dataset in threeClassData:
                
                # establish colors and colormap
                redish = '#d73027'
                greenish = '#50C878'
                blueish = '#4575b4'
                colormap = np.array([redish,blueish,greenish])

                #establish classes
                classes = ['0','1','2']
                plot_decision_boundary2D(model, pert.cpu().detach().numpy(), b.cpu().detach().numpy(), classes, colormap)
            else:
                plot_decision_boundary(lambda x: model(x), pert.cpu().detach().numpy(), b.cpu().detach().numpy())
            plt.savefig(data_img,dpi=resolution)
            plt.close()
            plot_attacks(data_img2, dataset, 'nnattack', test_loader, resolution)
        
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %d',
            epoch, train_time - start_time, lr_attack, train_loss/train_n, train_acc/train_n, train_n-train_acc)
        
        testac,fool_pgd,fool_fgsm,fool_nn,loss_pgd,loss_fgsm,loss_nn=Test(model, test_loader,'nn',epoch)
        pgd_acc,PGDfool_pgd,PGDfool_fgsm,PGDfool_nn,PGDloss_pgd,PGDloss_fgsm,PGDloss_nn=Test(pgd_model, test_loader,'pgd',epoch)
        if epoch ==0:
            pgdfgsm_buff=PGDfool_fgsm
            pgdpgd_buff=PGDfool_pgd
            losspgdfgsm_buff=PGDloss_fgsm
            losspgdpgd_buff=PGDloss_pgd
        acc=np.array((train_acc/train_n,testac))
        Acc.append(acc)
        fools.append(np.array((fool_pgd,fool_fgsm,fool_nn)))
        PGDfools.append(np.array((pgdpgd_buff,pgdfgsm_buff,PGDfool_nn)))
        LOSS.append(np.array((loss_pgd,loss_fgsm,loss_nn)))
        PGDLOSS.append(np.array((losspgdpgd_buff,losspgdfgsm_buff,PGDloss_nn)))
    print("***Training completed****")
    return Acc, fools, PGDfools,LOSS,PGDLOSS

def Test(model,test_loader,DEF,epoch):
    test_loss = 0
    test_acc = 0
    test_acc_pgd=0
    test_acc_fgsm=0
    test_acc_nn=0
    n = 0
    pgd_attack_loss=0
    fgsm_attack_loss=0
    nn_attack_loss=0
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        X=X.float()
        #y=y.float()
        y=y.type(torch.cuda.LongTensor)
        if DEF =='pgd' and epoch ==0:
            delta_pgd = attack_pgd(model, X, y, epsilon, 0.01, 50, 10,trim)
            delta_fgsm = attack_fgsm(model, X, y, epsilon,trim)
        elif DEF =='nn':
            delta_pgd = attack_pgd(model, X, y, epsilon, 0.01, 50, 10,trim)
            delta_fgsm = attack_fgsm(model, X, y, epsilon,trim)
        
        if lab:
            delta_nn = attack_net(X,y)
        else:
            delta_nn = attack_net(X)
        
        with torch.no_grad():
            if trim:
                output = model(torch.clamp(X,0,1))
            else:
                output = model(X)
            
            loss = criterion(output.squeeze(1), y) 
            
            test_loss += loss.item() * y.size(0)
            #out=(output.squeeze(1)>0.5).float()
            #test_acc += (out == y).sum().item()
            test_acc += (output.max(1)[1] == y).sum().item()
            if DEF =='pgd' and epoch ==0:
                test_acc_pgd += (model(X+delta_pgd).max(1)[1] == y).sum().item()
                test_acc_fgsm += (model(X+delta_fgsm).max(1)[1] == y).sum().item()
                
                pgd_loss= sum_criterion(model(X+delta_pgd).squeeze(1), y) # pgd attack loss
                fgsm_loss= sum_criterion(model(X+delta_fgsm).squeeze(1), y) # fgsm attack loss
                pgd_attack_loss += pgd_loss.item()
                fgsm_attack_loss += fgsm_loss.item()
                
            elif DEF =='nn':
                test_acc_pgd += (model(X+delta_pgd).max(1)[1] == y).sum().item()
                test_acc_fgsm += (model(X+delta_fgsm).max(1)[1] == y).sum().item()
                
                pgd_loss= sum_criterion(model(X+delta_pgd).squeeze(1), y) # pgd attack loss
                fgsm_loss= sum_criterion(model(X+delta_fgsm).squeeze(1), y) # fgsm attack loss
                pgd_attack_loss += pgd_loss.item()
                fgsm_attack_loss += fgsm_loss.item()
            
            test_acc_nn += (model(X+delta_nn).max(1)[1] == y).sum().item()
            nn_loss= sum_criterion(model(X+delta_nn).squeeze(1), y) # nn attack loss
            nn_attack_loss += nn_loss.item()
            n += y.size(0)
    logger.info('Test Loss and accuracy: \t %.4f \t %.4f', test_loss/n, test_acc/n)
      
    return test_acc/n, n-test_acc_pgd,n-test_acc_fgsm,n-test_acc_nn,pgd_attack_loss,fgsm_attack_loss,nn_attack_loss

def cw_loss(logits,labels,num_labels, adv_lambda):
    # cal adv loss
    probs_model = F.softmax(logits, dim=1)
    dev=probs_model.device
    onehot_labels = torch.eye(num_labels, device=dev)[labels]

    # C&W loss function
    real = torch.sum(onehot_labels * probs_model, dim=1)
    other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
    zeros = torch.zeros_like(other)
    loss_adv = torch.max(real - other, zeros)
    loss_adv = torch.sum(loss_adv)

    # maximize cross_entropy loss
    # loss_adv = -F.mse_loss(logits_model, onehot_labels)
    #loss_adv = - F.cross_entropy(logits_model, labels)

    return adv_lambda*loss_adv

"""Dataset"""  
seed=0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

dataset = 'circles'
input_path='Data/'+dataset+'.npy'
label_path='Data/'+dataset+'-label.npy'

covariate, response, train_data, test_data=load_dataset(dataset, input_path, label_path)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

signed_train=False
CW_loss=False
adv_lambda=10
P=2
resolution=300
#alpha_vals=np.arange(0.6,0.9,0.3)
alpha_vals=[0.6]
#eps_vals=np.arange(0.20,0.25,0.05)
eps_vals=np.array([0.15,0.20,0.25,0.30])

threeClassData=['streaks','blobs']

if dataset in threeClassData:
    N_cls=3 
else:
    N_cls=2

for epsilon in eps_vals:
    for alpha in alpha_vals:
        model=classifier(n_classes=N_cls).cuda()
        
       
        """ attack network"""
        lab=True
        epsilon=round(epsilon,3)
        if lab:
            attack_net = Attack_net_withlabelsLP_clamped(epsilon=epsilon, n_classes=N_cls,p=P,xdim=2).cuda()
            #attack_net = Attack_net_withlabelsLP(epsilon=epsilon, n_classes=N_cls,p=P).cuda()
            ind="lab"

        else:
            attack_net = Attack_net(epsilon=epsilon, n_classes=N_cls).cuda()
            ind="no-lab"



        """Adverserial training parameters"""
        trim= False
        nepochs=101
        pretrain_epochs=20
        burnin_epochs=5
        
        """ Loading previously adv trained pgd model"""
        pgd_model=classifier(n_classes=N_cls).cuda()
        fname='models/'+dataset+'-'+'pgd'+str(epsilon)+'.pth'
        #fname='models/'+'no_index'+dataset+'-'+'pgd'+str(epsilon)+'.pth' # for loss computation
        pgd_model.load_state_dict(torch.load(fname))
        



        alpha=round(alpha,2)
        headstart=1  #how much the attck network must be trained before defense updates of the model
        defense_iters=1 # no. of defense updates at once
        
        lr_classifier=1e-3
        lr_attack=2*1e-4#1e-3 #2*1e-4
        #epsilon=0.3

        lr_type='flat'
        """Training configuration"""
        pretrain=False
        burn_in=False # pretrain is necessary for burn in 
        pind='-pre-'if pretrain else ''
        bind = '-burn-'if burn_in else ''

        """ save path """
        defense='NNadv-'+ind+pind+bind
        dpath= 'Moon Data diagnosis/paper_models/'+dataset+'-'+defense+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
        apath= 'Moon Data diagnosis/paper_models/'+dataset+'-'+'advnn'+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'

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

        #criterion = nn.BCELoss() # binary cross entropy bernoulli likelihood loss
        criterion = nn.CrossEntropyLoss(reduction='mean')
        sum_criterion=nn.CrossEntropyLoss(reduction='sum')
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG)


        if pretrain:
            if burn_in:
                Pretrain(model, opt, pretrain_epochs)  
                opt = torch.optim.Adam(model.parameters(), lr=lr_classifier)
                opt_attack= torch.optim.Adam(attack_net.parameters(), lr=lr_attack)  
                BurnIn(model, attack_net, opt, opt_attack, burnin_epochs)
                opt = torch.optim.Adam(model.parameters(), lr=lr_classifier)
                opt_attack= torch.optim.Adam(attack_net.parameters(), lr=lr_attack)  
            else:
                Pretrain(model, opt, pretrain_epochs)
                opt = torch.optim.Adam(model.parameters(), lr=lr_classifier)
                opt_attack= torch.optim.Adam(attack_net.parameters(), lr=lr_attack) 
                
        AC,FLS, PGDFLS,LS,PGDLS=Train(model, attack_net, opt, opt_attack, nepochs, headstart,alpha)
        acc_array=np.vstack(AC)
        fool_array=np.vstack(FLS)
        pgdfool_array=np.vstack(PGDFLS)
        loss_array=np.vstack(LS)
        pgdloss_array=np.vstack(PGDLS)
        file_name='Moon Data diagnosis/paper_stats/'+dataset+"-"+str(alpha)+"-alpha-epsilon-"+str(epsilon)
        np.save(file_name+"ACC.npy",acc_array)
        np.save(file_name+"FOOL.npy",fool_array)
        np.save(file_name+"PGDFOOL.npy",pgdfool_array)
        
        #np.save(file_name+"LOSS.npy",loss_array)
        #np.save(file_name+"PGDLOSS.npy",pgdloss_array)
        
        """
        plot_name='Moon Data diagnosis/Plots/AccPlots/'+'L'+str(P)+' '+dataset+'-alpha-'+str(alpha)+'-epsilon-'+str(epsilon)
        
        plt.title(r'No. of fooled ($\alpha,\epsilon$)'+'=('+str(alpha)+','+str(epsilon)+') '+ind, fontsize='small')
        plt.plot(fool_array[:,0], color='green', label='def:nn atk:pgd')
        plt.plot(fool_array[:,1], color='blue', label='def:nn atk:fgsm')
        plt.plot(fool_array[:,2], color='red', label='def:nn atk:nn_adv')
        
        plt.plot(pgdfool_array[:,0], color='green',linestyle = ':', label='def:pgd atk:pgd',alpha=0.6)
        plt.plot(pgdfool_array[:,1], color='blue', linestyle = ':',label='def:pgd atk:fgsm',alpha=0.6)
        plt.plot(pgdfool_array[:,2], color='red',linestyle = ':', label='def:pgd atk:nn_adv',alpha=0.6)
        plt.xlabel('epochs')
        plt.ylabel('Fooled')
        plt.legend(fontsize='x-small')
        plt.savefig(plot_name+'-'+ind+'-fools.png',dpi=300)
        plt.close()
        
        plt.title('Train vs Test accuracy-'+ind, fontsize='small')
        plt.plot(acc_array[:,0], color='green', label='Train')
        plt.plot(acc_array[:,1], color='blue', label='Test')
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.legend(fontsize='x-small')
        plt.savefig(plot_name+'-'+ind+'-accs.png',dpi=300)
        plt.close()
        """
        torch.save(model.state_dict(), dpath)
        torch.save(attack_net.state_dict(),apath)
