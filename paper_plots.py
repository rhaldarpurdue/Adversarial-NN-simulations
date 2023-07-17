# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:30:56 2022

@author: rajde
"""
import torch
from dataset import load_dataset
from net import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator

epsilon=0.2
alpha=0.0
P=2
ind="lab"
pind=""
bind=""
dataset='moon'
defense='NNadv-'+ind+pind+bind
#dpath='models/'+dataset+'-none0.3.pth'
dpath= 'Moon Data diagnosis/paper_loss/'+dataset+'-'+defense+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
apath= 'Moon Data diagnosis/paper_loss/'+dataset+'-'+'advnn'+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'


input_path='Data/'+dataset+'.npy'
label_path='Data/'+dataset+'-label.npy'

covariate, response, train_data, test_data=load_dataset(dataset, input_path, label_path)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

threeClassData=['streaks','blobs']

if dataset in threeClassData:
    N_cls=3 
else:
    N_cls=2

attack_model=Attack_net_withlabelsLP_clamped(epsilon=epsilon, n_classes=N_cls,p=P,xdim=2)
defense_model=classifier(n_classes=N_cls)

attack_model.load_state_dict(torch.load(apath))
defense_model.load_state_dict(torch.load(dpath))

#----------Grad plot-----------------
#contour plot
def grad_plot(att=True):
    plt.style.use('plot_style.txt')
    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)#scientific notation
    xl, yl = covariate.min(axis=0) - 0.1
    xu, yu = covariate.max(axis=0) + 0.1

    aa, bb = np.mgrid[xl:xu:100j, yl:yu:100j]
    ab = np.c_[aa.ravel(), bb.ravel()]

    ab_T = torch.tensor(ab).float()
    z=defense_model(ab_T)
    y_pred=torch.max(z,dim=1).indices
    #data_est=torch.cat((ab_T,y_pred.unsqueeze(1)),dim=1)

    criterion = nn.CrossEntropyLoss(reduction='none')
    loss=criterion(z,y_pred) #loss at each data point
    loss_sum=torch.sum(loss)

    c=loss.detach().to('cpu').numpy()
    cc = c.reshape(aa.shape)

    plt.gca().patch.set_color('#440154')
    cp = plt.contourf(aa,bb,cc,locator=ticker.LogLocator(),extend='both') #use log-scale
    #cp = plt.contourf(aa,bb,cc,extend='both')
    cb = plt.colorbar(cp,format=ticker.FuncFormatter(fmt))

    aa, bb = np.mgrid[xl:xu:20j, yl:yu:20j]
    ab = np.c_[aa.ravel(), bb.ravel()]

    ab_T = torch.tensor(ab).float()
    X_clone=torch.clone(ab_T)
    X_clone.requires_grad=True
    z=defense_model(X_clone)
    y_pred=torch.max(z,dim=1).indices
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss=criterion(z,y_pred) #loss at each data point
    loss_sum=torch.sum(loss)
    loss_sum.backward()
    if(att):
        UV=attack_model(ab_T,y_pred).detach().to('cpu').numpy()
        print("attack model plot")
    else:
        UV=X_clone.grad.detach().to('cpu').numpy()
        print("grad plot")
    U=UV[:,0]
    uu=U.reshape(aa.shape)
    V=UV[:,1]
    vv=V.reshape(aa.shape)
    speed = np.sqrt(uu**2 + vv**2)
    UN = uu/speed
    VN = vv/speed
    if(att):
        col="magenta"
        plt.title(r'$\ell_{\infty}(0.2)$')
    else:
        col="Red"
        plt.title(r'$\nabla \mathcal{L}$')
    quiv = plt.quiver(aa, bb, UN, VN,  # assign to var
               color=col, 
               headlength=7)

    plt.xlim([xl+0.1,xu-0.1])
    plt.ylim([yl,yu])
    #plt.show()
    plt.savefig('paper_plots/grad.png')


#grad_plot(att=False)

#-------Performance plot-----------

def perf_plot():
    plt.style.use('plot_style.txt')
    file_name='Moon Data diagnosis/paper_stats/'+dataset+"-"+str(0.0)+"-alpha-epsilon-"+str(epsilon)
    acc_array=np.load(file_name+"ACC.npy")
    fool_array=np.load(file_name+"FOOL.npy")
    pgdfool_array=np.load(file_name+"PGDFOOL.npy")
    
    file_name='Moon Data diagnosis/paper_stats/'+dataset+"-"+str(0.3)+"-alpha-epsilon-"+str(epsilon)
    acc_array2=np.load(file_name+"ACC.npy")
    fool_array2=np.load(file_name+"FOOL.npy")
    pgdfool_array2=np.load(file_name+"PGDFOOL.npy")
    
    lwd=0.8
    lwd2=1.5
    alp=0.7

    plt.title(dataset+r'$(\delta=0.2)$')
    plt.plot(fool_array[:,0], color='green', label=r'$f,\lambda_{PGD},\alpha=0$',linewidth=lwd,alpha=alp)
    plt.plot(fool_array[:,1], color='blue', label=r'$f,\lambda_{FGSM},\alpha=0$',linewidth=lwd,alpha=alp)
    plt.plot(fool_array[:,2], color='red', label=r'$f,\lambda^f,\alpha=0$',linewidth=lwd,alpha=alp)

    plt.plot(pgdfool_array[:,0], color='green',linestyle = 'dashdot', label=r'$f_{PGD},\lambda_{PGD}$',alpha=0.8,linewidth=lwd2)
    plt.plot(pgdfool_array[:,1], color='blue', linestyle = ':',label=r'$f_{PGD},\lambda_{FGSM}$',alpha=0.8,linewidth=lwd2)
    plt.plot(pgdfool_array[:,2], color='red',linestyle = ':', label=r'$f_{PGD},\lambda^f,\alpha=0$',alpha=0.8,linewidth=lwd2)
    
    plt.plot(fool_array2[:,0], color='orange', label=r'$f,\lambda_{PGD},\alpha=0.3$',linewidth=lwd,alpha=alp)
    plt.plot(fool_array2[:,1], color='cyan', label=r'$f,\lambda_{FGSM},\alpha=0.3$',linewidth=lwd,alpha=alp)
    plt.plot(fool_array2[:,2], color='magenta', label=r'$f,\lambda^f,\alpha=0.3$',linewidth=lwd,alpha=alp)

    #plt.plot(pgdfool_array2[:,0], color='yellow',linestyle = ':', label=r'$f_{PGD},\lambda_{PGD},\alpha=0.3$',alpha=0.6,linewidth=lwd)
    #plt.plot(pgdfool_array2[:,1], color='cyan', linestyle = ':',label=r'$f_{PGD},\lambda_{FGSM},\alpha=0.3$',alpha=0.6,linewidth=lwd)
    plt.plot(pgdfool_array2[:,2], color='magenta',linestyle = ':', label=r'$f_{PGD},\lambda^f,\alpha=0.3$',alpha=0.8,linewidth=lwd2)
    plt.xlabel(r'$T$')
    plt.ylabel(r'Fooled')
    plt.legend(fontsize='x-small',ncol=2,loc='upper right')
    yu=np.max(fool_array[2:100,:])
    yu2=np.max(fool_array2[2:100,:])
    plt.xlim([0,100])
    plt.ylim([0,max(yu,yu2)])
    plt.savefig('paper_plots/'+dataset+'.png',dpi=300)

#perf_plot()
#grad_plot(att=True)

def grid_perf_plot():
    DATA=['circles','moon','polynomial','streaks']
    plt.style.use('plot_style.txt')
    fig, axs = plt.subplots(2, 2,sharex='col',figsize=(10,6))
    lwd=0.8
    lwd2=1.5
    alp=0.7
    for dataset, ax in zip(DATA, axs.ravel()):
        file_name='Moon Data diagnosis/paper_stats/'+dataset+"-"+str(0.0)+"-alpha-epsilon-"+str(epsilon)
        if dataset=='polynomial':
            test_samples=900
        else:
            test_samples=500
        acc_array=np.load(file_name+"ACC.npy")
        fool_array=np.load(file_name+"FOOL.npy")*100/test_samples
        pgdfool_array=np.load(file_name+"PGDFOOL.npy")*100/test_samples
        
        """file_name='Moon Data diagnosis/paper_stats/'+dataset+"-"+str(0.3)+"-alpha-epsilon-"+str(epsilon)
        acc_array2=np.load(file_name+"ACC.npy")
        fool_array2=np.load(file_name+"FOOL.npy")*100/test_samples
        pgdfool_array2=np.load(file_name+"PGDFOOL.npy")*100/test_samples"""
        ax.set_title(dataset)
        ax.plot(fool_array[:,0], color='green', label=r'$f,\lambda_{PGD}$',linewidth=lwd,alpha=alp)
        ax.plot(fool_array[:,1], color='blue', label=r'$f,\lambda_{FGSM}$',linewidth=lwd,alpha=alp)
        ax.plot(fool_array[:,2], color='red', label=r'$f,\lambda^f$',linewidth=lwd,alpha=alp)

        ax.plot(pgdfool_array[:,0], color='green',linestyle = 'dashdot', label=r'$f_{PGD},\lambda_{PGD}$',alpha=0.8,linewidth=lwd2)
        ax.plot(pgdfool_array[:,1], color='blue', linestyle = ':',label=r'$f_{PGD},\lambda_{FGSM}$',alpha=0.8,linewidth=lwd2)
        ax.plot(pgdfool_array[:,2], color='red',linestyle = ':', label=r'$f_{PGD},\lambda^f$',alpha=0.8,linewidth=lwd2)
        
        """ax.plot(fool_array2[:,0], color='orange', label=r'$f,\lambda_{PGD},\alpha=0.3$',linewidth=lwd,alpha=alp)
        ax.plot(fool_array2[:,1], color='cyan', label=r'$f,\lambda_{FGSM},\alpha=0.3$',linewidth=lwd,alpha=alp)
        ax.plot(fool_array2[:,2], color='magenta', label=r'$f,\lambda^f,\alpha=0.3$',linewidth=lwd,alpha=alp)

        #plt.plot(pgdfool_array2[:,0], color='yellow',linestyle = ':', label=r'$f_{PGD},\lambda_{PGD},\alpha=0.3$',alpha=0.6,linewidth=lwd)
        #plt.plot(pgdfool_array2[:,1], color='cyan', linestyle = ':',label=r'$f_{PGD},\lambda_{FGSM},\alpha=0.3$',alpha=0.6,linewidth=lwd)
        ax.plot(pgdfool_array2[:,2], color='magenta',linestyle = ':', label=r'$f_{PGD},\lambda^f,\alpha=0.3$',alpha=0.8,linewidth=lwd2)"""
        yu=np.max(fool_array[2:100,:])
        yu2=np.max(pgdfool_array[2:100,:])
        ax.set_xlim([0,100])
        ax.set_ylim([0,max(yu,yu2)])
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
    axs[0,0].set_ylabel('Fooled %')
    axs[1,0].set_ylabel('Fooled %')
    axs[1,0].set_xlabel(r'$T$')
    axs[1,1].set_xlabel(r'$T$')
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2)
    fig.suptitle(r'$\ell_{\infty}(0.2)$ VS PGD')
    plt.savefig('paper_plots/perf_plot.png',dpi=600)

#grid_perf_plot()

def grid_clean_plot():
    alpha=0.0
    epsilon=0.2
    DATA=['circles','moon','polynomial','streaks']
    plt.style.use('plot_style.txt')
    
    fig, axsLeft = plt.subplots(len(DATA),figsize=(5,15))

    att=False
    for i,dataset in enumerate(DATA):
        input_path='Data/'+dataset+'.npy'
        label_path='Data/'+dataset+'-label.npy'
        covariate, response, train_data, test_data=load_dataset(dataset, input_path, label_path)
        dpath='models/'+dataset+'-none.pth'
        #dpath='Moon Data diagnosis/paper_models/'+dataset+'-'+defense+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
        apath= 'Moon Data diagnosis/paper_models/'+dataset+'-'+'advnn'+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
        threeClassData=['streaks','blobs']
        if dataset in threeClassData:
            N_cls=3 
        else:
            N_cls=2

        attack_model=Attack_net_withlabelsLP_clamped(epsilon=epsilon, n_classes=N_cls,p=P,xdim=2)
        defense_model=classifier(n_classes=N_cls)
        attack_model.load_state_dict(torch.load(apath))
        defense_model.load_state_dict(torch.load(dpath))
        
        def fmt(x, pos):
            a, b = '{:.2e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)#scientific notation
        xl, yl = covariate.min(axis=0) - 0.1
        xu, yu = covariate.max(axis=0) + 0.1

        aa, bb = np.mgrid[xl:xu:100j, yl:yu:100j]
        ab = np.c_[aa.ravel(), bb.ravel()]

        ab_T = torch.tensor(ab).float()
        z=defense_model(ab_T)
        y_pred=torch.max(z,dim=1).indices
        #data_est=torch.cat((ab_T,y_pred.unsqueeze(1)),dim=1)

        criterion = nn.CrossEntropyLoss(reduction='none')
        loss=criterion(z,y_pred) #loss at each data point
        loss_sum=torch.sum(loss)

        c=loss.detach().to('cpu').numpy()
        cc = c.reshape(aa.shape)

        axsLeft[i].patch.set_color('#440154')
        if dataset in ['moon','streaks']:
            cp = axsLeft[i].contourf(aa,bb,cc,locator=ticker.LogLocator(),extend='both') #use log-scale
        else:
            cp = axsLeft[i].contourf(aa,bb,cc,extend='both')
        #cb = plt.colorbar(cp,format=ticker.FuncFormatter(fmt))

        aa, bb = np.mgrid[xl:xu:20j, yl:yu:20j]
        ab = np.c_[aa.ravel(), bb.ravel()]

        ab_T = torch.tensor(ab).float()
        X_clone=torch.clone(ab_T)
        X_clone.requires_grad=True
        z=defense_model(X_clone)
        y_pred=torch.max(z,dim=1).indices
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss=criterion(z,y_pred) #loss at each data point
        loss_sum=torch.sum(loss)
        loss_sum.backward()
        if(att):
            UV=attack_model(ab_T,y_pred).detach().to('cpu').numpy()
            print("attack model plot")
        else:
            UV=X_clone.grad.detach().to('cpu').numpy()
            print("grad plot")
        if(att):
            col="magenta"
        else:
            col="Red"
        U=UV[:,0]
        uu=U.reshape(aa.shape)
        V=UV[:,1]
        vv=V.reshape(aa.shape)
        speed = np.sqrt(uu**2 + vv**2)
        UN = uu/speed
        VN = vv/speed
        axsLeft[i].quiver(aa, bb, UN, VN,  # assign to var
                   color=col, 
                   headlength=7)

        axsLeft[i].set_xlim([xl+0.1,xu-0.1])
        axsLeft[i].set_ylim([yl,yu])
        axsLeft[i].set_ylabel(dataset)
        
        
    #fig.suptitle(r'$(a)$ Clean Train')
    axsLeft[0].set_title(r'$\nabla_{x} \mathcal{L}$',loc='center')
    plt.savefig('paper_plots/clean_train2.png',dpi=600)

#grid_clean_plot()


def grid_grad_plot():
    alpha=0.0
    epsilon=0.2
    DATA=['circles','moon','polynomial','streaks']
    plt.style.use('plot_style.txt')
    l='infinity'
    if l=='infinity':
        fig, axsLeft = plt.subplots(len(DATA),2,figsize=(10,15),sharey='row',gridspec_kw={'width_ratios': [0.8, 1]})
    else:
        fig, axsLeft = plt.subplots(len(DATA),2,figsize=(9,15),sharey='row')
    plt.subplots_adjust(wspace=0.01)
    for att in [False,True]:
        for i,dataset in enumerate(DATA):
            input_path='Data/'+dataset+'.npy'
            label_path='Data/'+dataset+'-label.npy'
            covariate, response, train_data, test_data=load_dataset(dataset, input_path, label_path)
            if l=='infinity':
                dpath='Moon Data diagnosis/paper_models/'+dataset+'-'+defense+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
                apath= 'Moon Data diagnosis/paper_models/'+dataset+'-'+'advnn'+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
            else:
                dpath='Moon Data diagnosis/paper_models/'+'L2'+dataset+'-'+defense+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
                apath= 'Moon Data diagnosis/paper_models/'+'L2'+dataset+'-'+'advnn'+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
            threeClassData=['streaks','blobs']
            if dataset in threeClassData:
                N_cls=3 
            else:
                N_cls=2

            if l=='infinity':
                attack_model=Attack_net_withlabelsLP_clamped(epsilon=epsilon, n_classes=N_cls,p=P,xdim=2)
            else:
                attack_model=Attack_net_withlabelsLP(epsilon=epsilon, n_classes=N_cls,p=P)
            defense_model=classifier(n_classes=N_cls)
            attack_model.load_state_dict(torch.load(apath))
            defense_model.load_state_dict(torch.load(dpath))
            
            def fmt(x, pos):
                a, b = '{:.2e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)#scientific notation
            xl, yl = covariate.min(axis=0) - 0.1
            xu, yu = covariate.max(axis=0) + 0.1

            aa, bb = np.mgrid[xl:xu:100j, yl:yu:100j]
            ab = np.c_[aa.ravel(), bb.ravel()]

            ab_T = torch.tensor(ab).float()
            z=defense_model(ab_T)
            y_pred=torch.max(z,dim=1).indices
            #data_est=torch.cat((ab_T,y_pred.unsqueeze(1)),dim=1)

            criterion = nn.CrossEntropyLoss(reduction='none')
            loss=criterion(z,y_pred) #loss at each data point
            loss_sum=torch.sum(loss)

            c=loss.detach().to('cpu').numpy()
            cc = c.reshape(aa.shape)

            axsLeft[i,int(att)].patch.set_color('#440154') # set patch color to zero
            if dataset in ['moon','streaks']:
                cp = axsLeft[i,int(att)].contourf(aa,bb,cc,locator=ticker.LogLocator(),extend='both') #use log-scale
            else:
                cp = axsLeft[i,int(att)].contourf(aa,bb,cc,extend='both')
                
            if att and l=='infinity':
                cb = plt.colorbar(cp,format=ticker.FuncFormatter(fmt),ax=axsLeft[i,int(att)])

            aa, bb = np.mgrid[xl:xu:20j, yl:yu:20j]
            ab = np.c_[aa.ravel(), bb.ravel()]

            ab_T = torch.tensor(ab).float()
            X_clone=torch.clone(ab_T)
            X_clone.requires_grad=True
            z=defense_model(X_clone)
            y_pred=torch.max(z,dim=1).indices
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss=criterion(z,y_pred) #loss at each data point
            loss_sum=torch.sum(loss)
            loss_sum.backward()
            if(att):
                UV=attack_model(ab_T,y_pred).detach().to('cpu').numpy()
                print("attack model plot")
            else:
                UV=X_clone.grad.detach().to('cpu').numpy()
                print("grad plot")
            if(att):
                col="magenta"
            else:
                col="Red"
            U=UV[:,0]
            uu=U.reshape(aa.shape)
            V=UV[:,1]
            vv=V.reshape(aa.shape)
            speed = np.sqrt(uu**2 + vv**2)
            UN = uu/speed
            VN = vv/speed
            axsLeft[i,int(att)].quiver(aa, bb, UN, VN,  # assign to var
                       color=col, 
                       headlength=7)

            axsLeft[i,int(att)].set_xlim([xl+0.1,xu-0.1])
            axsLeft[i,int(att)].set_ylim([yl,yu])
        
        
    #fig.suptitle(r'$(a)$ Clean Train')
    axsLeft[0,0].set_title(r'$\nabla_{x} \mathcal{L}$',loc='center')
    axsLeft[0,1].set_title(r'$\lambda^f$',loc='center')
    plt.savefig('paper_plots/Linf_grad2.png',dpi=600)

#grid_grad_plot()
def grid_loss_plot():
    DATA=['circles','moon','polynomial','streaks']
    plt.style.use('plot_style.txt')
    fig, axs = plt.subplots(2, 2,sharex='col',figsize=(10,6))
    lwd=0.8
    lwd2=1.5
    alp=0.7
    for dataset, ax in zip(DATA, axs.ravel()):
        file_name='Moon Data diagnosis/paper_loss/'+dataset+"-"+str(0.0)+"-alpha-epsilon-"+str(epsilon)
        acc_array=np.load(file_name+"ACC.npy")
        fool_array=np.load(file_name+"LOSS.npy")
        pgdfool_array=np.load(file_name+"PGDLOSS.npy")
        
        """file_name='Moon Data diagnosis/paper_stats/'+dataset+"-"+str(0.3)+"-alpha-epsilon-"+str(epsilon)
        acc_array2=np.load(file_name+"ACC.npy")
        fool_array2=np.load(file_name+"FOOL.npy")*100/test_samples
        pgdfool_array2=np.load(file_name+"PGDFOOL.npy")*100/test_samples"""
        ax.set_title(dataset)
        ax.plot(fool_array[:,0], color='green', label=r'$f,\lambda_{PGD}$',linewidth=lwd,alpha=alp)
        ax.plot(fool_array[:,1], color='blue', label=r'$f,\lambda_{FGSM}$',linewidth=lwd,alpha=alp)
        ax.plot(fool_array[:,2], color='red', label=r'$f,\lambda^f$',linewidth=lwd,alpha=alp)

        ax.plot(pgdfool_array[:,0], color='green',linestyle = 'dashdot', label=r'$f_{PGD},\lambda_{PGD}$',alpha=0.8,linewidth=lwd2)
        ax.plot(pgdfool_array[:,1], color='blue', linestyle = ':',label=r'$f_{PGD},\lambda_{FGSM}$',alpha=0.8,linewidth=lwd2)
        ax.plot(pgdfool_array[:,2], color='red',linestyle = ':', label=r'$f_{PGD},\lambda^f$',alpha=0.8,linewidth=lwd2)
        
        """ax.plot(fool_array2[:,0], color='orange', label=r'$f,\lambda_{PGD},\alpha=0.3$',linewidth=lwd,alpha=alp)
        ax.plot(fool_array2[:,1], color='cyan', label=r'$f,\lambda_{FGSM},\alpha=0.3$',linewidth=lwd,alpha=alp)
        ax.plot(fool_array2[:,2], color='magenta', label=r'$f,\lambda^f,\alpha=0.3$',linewidth=lwd,alpha=alp)

        #plt.plot(pgdfool_array2[:,0], color='yellow',linestyle = ':', label=r'$f_{PGD},\lambda_{PGD},\alpha=0.3$',alpha=0.6,linewidth=lwd)
        #plt.plot(pgdfool_array2[:,1], color='cyan', linestyle = ':',label=r'$f_{PGD},\lambda_{FGSM},\alpha=0.3$',alpha=0.6,linewidth=lwd)
        ax.plot(pgdfool_array2[:,2], color='magenta',linestyle = ':', label=r'$f_{PGD},\lambda^f,\alpha=0.3$',alpha=0.8,linewidth=lwd2)"""
        yu=np.max(fool_array[2:100,:])
        yu2=np.max(pgdfool_array[2:100,:])
        ax.set_xlim([0,100])
        ax.set_ylim([0,max(yu,yu2)])
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
    axs[0,0].set_ylabel(r'$\mathcal{L}$')
    axs[1,0].set_ylabel(r'$\mathcal{L}$')
    axs[1,0].set_xlabel(r'$T$')
    axs[1,1].set_xlabel(r'$T$')
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2)
    fig.suptitle(r'$\ell_{\infty}(0.2)$ VS PGD')
    plt.savefig('paper_plots/loss_plot.png',dpi=600)
    
#grid_loss_plot()

def reg_loss_plot():
    epsilon=0.7
    alpha=0.0
    plt.style.use('plot_style.txt')
    dataset='boston'
    file_name='Moon Data diagnosis/paper_reg/'+dataset+"-"+str(0.0)+"-alpha-epsilon-"+str(epsilon)
    fool_array=np.load(file_name+"LOSS.npy")
    pgdfool_array=np.load(file_name+"PGDLOSS.npy")
    
    lwd=0.8
    lwd2=1.5
    alp=0.7

    plt.title(dataset+r'$(\delta=0.7)$')
    plt.plot(fool_array[:,0], color='green', label=r'$f,\lambda_{PGD}$',linewidth=lwd,alpha=alp)
    plt.plot(fool_array[:,1], color='blue', label=r'$f,\lambda_{FGSM}$',linewidth=lwd,alpha=alp)
    plt.plot(fool_array[:,2], color='red', label=r'$f,\lambda^f$',linewidth=lwd,alpha=alp)

    plt.plot(pgdfool_array[:,0], color='green',linestyle = 'dashdot', label=r'$f_{PGD},\lambda_{PGD}$',alpha=0.8,linewidth=lwd2)
    plt.plot(pgdfool_array[:,1], color='blue', linestyle = ':',label=r'$f_{PGD},\lambda_{FGSM}$',alpha=0.8,linewidth=lwd2)
    plt.plot(pgdfool_array[:,2], color='red',linestyle = ':', label=r'$f_{PGD},\lambda^f$',alpha=0.8,linewidth=lwd2)
    
    plt.xlabel(r'$T$')
    plt.ylabel(r'$\mathcal{L}$')
    plt.legend(fontsize='x-small',ncol=2,loc='upper right')
    yu=np.max(fool_array[75:400,:])
    yu2=np.max(pgdfool_array[75:400,:])
    plt.xlim([0,400])
    plt.ylim([0,max(yu,yu2)])
    plt.show()
    #plt.savefig('paper_plots/'+dataset+'.png',dpi=300)
    

def grid_reg_loss_plot():
    epsilon=0.2
    alpha=0.0
    DATA=['boston','diabetes']
    plt.style.use('plot_style.txt')
    fig, axs = plt.subplots(1, 2,sharex='col',figsize=(10,4))
    lwd=0.8
    lwd2=1.5
    alp=0.7
    for dataset, ax in zip(DATA, axs.ravel()):
        file_name='Moon Data diagnosis/paper_reg/'+dataset+"-"+str(0.0)+"-alpha-epsilon-"+str(epsilon)
        fool_array=np.load(file_name+"LOSS.npy")
        pgdfool_array=np.load(file_name+"PGDLOSS.npy")
        
        ax.set_title(dataset,loc='center')
        ax.plot(fool_array[:,0], color='green', label=r'$f,\lambda_{PGD}$',linewidth=lwd,alpha=alp)
        ax.plot(fool_array[:,1], color='blue', label=r'$f,\lambda_{FGSM}$',linewidth=lwd,alpha=alp)
        ax.plot(fool_array[:,2], color='red', label=r'$f,\lambda^f$',linewidth=lwd,alpha=alp)

        ax.plot(pgdfool_array[:,0], color='green',linestyle = 'dashdot', label=r'$f_{PGD},\lambda_{PGD}$',alpha=0.8,linewidth=lwd2)
        ax.plot(pgdfool_array[:,1], color='blue', linestyle = ':',label=r'$f_{PGD},\lambda_{FGSM}$',alpha=0.8,linewidth=lwd2)
        ax.plot(pgdfool_array[:,2], color='red',linestyle = ':', label=r'$f_{PGD},\lambda^f$',alpha=0.8,linewidth=lwd2)
        

        yu=np.max(fool_array[2:400,:])
        yu2=np.max(pgdfool_array[2:400,:])
        ax.set_xlim([0,400])
        ax.set_ylim([0,max(yu,yu2)])
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
    axs[0].set_ylabel(r'$\mathcal{L}$')
    axs[0].set_xlabel(r'$T$')
    axs[1].set_xlabel(r'$T$')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2)
    fig.suptitle(r'$\ell_{2}(0.7)$ VS PGD')
    plt.subplots_adjust(top=0.8) 
    plt.savefig('paper_plots/loss_reg_plot.png',dpi=600)

#grid_reg_loss_plot()

def dataset_plot():
    DATA=['circles','polynomial','moon','streaks']
    plt.style.use('plot_style.txt')
    fig, axs = plt.subplots(2, 2,figsize=(7,4))
    for dataset, ax in zip(DATA, axs.ravel()):
        input_path='Data/'+dataset+'.npy'
        label_path='Data/'+dataset+'-label.npy'
        X, Y, _, _=load_dataset(dataset, input_path, label_path)
        cdict = {0: 'red', 1: 'blue', 2: 'green'}
        for g in np.unique(Y):
            ix = np.where(Y == g)
            ax.scatter(X[ix,0], X[ix,1], c = cdict[g], label = g,s=20)
            ax.set_title(dataset)
    handles, labels = axs[1,1].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=1,title='Classes')
    fig.suptitle(r'Simulated datasets')
    plt.subplots_adjust(hspace=0.4,top=0.85)
    plt.savefig('paper_plots/dataset.png',dpi=600)
    
def alpha_plot():
    alpha=0.0
    epsilon=0.25
    P=2
    DATA=['circles','circles']
    plt.style.use('plot_style.txt')
    l='infinity'
    if l=='infinity':
        fig, axsLeft = plt.subplots(1,2,figsize=(10,4),sharey='row',gridspec_kw={'width_ratios': [0.8, 1]})
    else:
        fig, axsLeft = plt.subplots(1,2,figsize=(9,4),sharey='row')
    plt.subplots_adjust(wspace=0.01)
    for att in [True]:
        for i,dataset in enumerate(DATA):
            input_path='Data/'+dataset+'.npy'
            label_path='Data/'+dataset+'-label.npy'
            if i==1:
                alpha=0.6
            covariate, response, train_data, test_data=load_dataset(dataset, input_path, label_path)
            if l=='infinity':
                dpath='Moon Data diagnosis/paper_models/'+dataset+'-'+defense+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
                apath= 'Moon Data diagnosis/paper_models/'+dataset+'-'+'advnn'+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
            else:
                dpath='Moon Data diagnosis/paper_models/'+'L2'+dataset+'-'+defense+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
                apath= 'Moon Data diagnosis/paper_models/'+'L2'+dataset+'-'+'advnn'+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
            threeClassData=['streaks','blobs']
            if dataset in threeClassData:
                N_cls=3 
            else:
                N_cls=2

            if l=='infinity':
                attack_model=Attack_net_withlabelsLP_clamped(epsilon=epsilon, n_classes=N_cls,p=P,xdim=2)
            else:
                attack_model=Attack_net_withlabelsLP(epsilon=epsilon, n_classes=N_cls,p=P)
            defense_model=classifier(n_classes=N_cls)
            attack_model.load_state_dict(torch.load(apath))
            defense_model.load_state_dict(torch.load(dpath))
            
            def fmt(x, pos):
                a, b = '{:.2e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)#scientific notation
            xl, yl = covariate.min(axis=0) - 0.1
            xu, yu = covariate.max(axis=0) + 0.1

            aa, bb = np.mgrid[xl:xu:100j, yl:yu:100j]
            ab = np.c_[aa.ravel(), bb.ravel()]

            ab_T = torch.tensor(ab).float()
            z=defense_model(ab_T)
            y_pred=torch.max(z,dim=1).indices
            #data_est=torch.cat((ab_T,y_pred.unsqueeze(1)),dim=1)

            criterion = nn.CrossEntropyLoss(reduction='none')
            loss=criterion(z,y_pred) #loss at each data point
            loss_sum=torch.sum(loss)

            c=loss.detach().to('cpu').numpy()
            cc = c.reshape(aa.shape)

            axsLeft[i].patch.set_color('#440154') # set patch color to zero
            if dataset in ['moon','streaks']:
                cp = axsLeft[i].contourf(aa,bb,cc,locator=ticker.LogLocator(),extend='both') #use log-scale
            else:
                cp = axsLeft[i].contourf(aa,bb,cc,extend='both')
                
            if i==1:
                cb = plt.colorbar(cp,format=ticker.FuncFormatter(fmt),ax=axsLeft[i])

            aa, bb = np.mgrid[xl:xu:20j, yl:yu:20j]
            ab = np.c_[aa.ravel(), bb.ravel()]

            ab_T = torch.tensor(ab).float()
            X_clone=torch.clone(ab_T)
            X_clone.requires_grad=True
            z=defense_model(X_clone)
            y_pred=torch.max(z,dim=1).indices
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss=criterion(z,y_pred) #loss at each data point
            loss_sum=torch.sum(loss)
            loss_sum.backward()
            if(att):
                UV=attack_model(ab_T,y_pred).detach().to('cpu').numpy()
                print("attack model plot")
            else:
                UV=X_clone.grad.detach().to('cpu').numpy()
                print("grad plot")
            if(att):
                col="magenta"
            else:
                col="Red"
            U=UV[:,0]
            uu=U.reshape(aa.shape)
            V=UV[:,1]
            vv=V.reshape(aa.shape)
            speed = np.sqrt(uu**2 + vv**2)
            UN = uu/speed
            VN = vv/speed
            axsLeft[i].quiver(aa, bb, UN, VN,  # assign to var
                       color=col, 
                       headlength=7)

            axsLeft[i].set_xlim([xl+0.1,xu-0.1])
            axsLeft[i].set_ylim([yl,yu])
    axsLeft[0].set_title(r'$\lambda^f,\alpha=0$')
    axsLeft[1].set_title(r'$\lambda^f,\alpha=0.3$')
    plt.savefig('paper_plots/alpha_compare0.3.png',dpi=600)

def alpha_grid_plot():
    alpha_val=[0.0,0.3,0.6]
    eps_val=[0.15,0.20,0.25,0.30]
    P=2
    dataset='circles'
    l='infinity'
    plt.style.use('plot_style.txt')
    fig, axsLeft = plt.subplots(4,3,figsize=(7,10),sharey='row',sharex='col')
    plt.subplots_adjust(wspace=0.01)
    plt.subplots_adjust(hspace=0.01)
    att=True
    for i,epsilon in enumerate(eps_val):
        for j,alpha in enumerate(alpha_val):
            input_path='Data/'+dataset+'.npy'
            label_path='Data/'+dataset+'-label.npy'
            covariate, response, train_data, test_data=load_dataset(dataset, input_path, label_path)
            if l=='infinity':
                dpath='Moon Data diagnosis/paper_models/'+dataset+'-'+defense+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
                apath= 'Moon Data diagnosis/paper_models/'+dataset+'-'+'advnn'+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
            else:
                dpath='Moon Data diagnosis/paper_models/'+'L2'+dataset+'-'+defense+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
                apath= 'Moon Data diagnosis/paper_models/'+'L2'+dataset+'-'+'advnn'+'ep-'+str(epsilon)+'alpha-'+str(alpha)+'.pth'
            threeClassData=['streaks','blobs']
            if dataset in threeClassData:
                N_cls=3 
            else:
                N_cls=2

            if l=='infinity':
                attack_model=Attack_net_withlabelsLP_clamped(epsilon=epsilon, n_classes=N_cls,p=P,xdim=2)
            else:
                attack_model=Attack_net_withlabelsLP(epsilon=epsilon, n_classes=N_cls,p=P)
            defense_model=classifier(n_classes=N_cls)
            attack_model.load_state_dict(torch.load(apath))
            defense_model.load_state_dict(torch.load(dpath))
            
            def fmt(x, pos):
                a, b = '{:.2e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)#scientific notation
            xl, yl = covariate.min(axis=0) - 0.1
            xu, yu = covariate.max(axis=0) + 0.1

            aa, bb = np.mgrid[xl:xu:100j, yl:yu:100j]
            ab = np.c_[aa.ravel(), bb.ravel()]

            ab_T = torch.tensor(ab).float()
            z=defense_model(ab_T)
            y_pred=torch.max(z,dim=1).indices
            #data_est=torch.cat((ab_T,y_pred.unsqueeze(1)),dim=1)

            criterion = nn.CrossEntropyLoss(reduction='none')
            loss=criterion(z,y_pred) #loss at each data point
            loss_sum=torch.sum(loss)

            c=loss.detach().to('cpu').numpy()
            cc = c.reshape(aa.shape)

            axsLeft[i,j].patch.set_color('#440154') # set patch color to zero
            if dataset in ['moon','streaks']:
                cp = axsLeft[i,j].contourf(aa,bb,cc,locator=ticker.LogLocator(),extend='both') #use log-scale
            else:
                cp = axsLeft[i,j].contourf(aa,bb,cc,extend='both')
            
            #if i==0 and j==2:
                #cb = plt.colorbar(cp,format=ticker.FuncFormatter(fmt),ax=axsLeft[:,j],location='right')

            aa, bb = np.mgrid[xl:xu:20j, yl:yu:20j]
            ab = np.c_[aa.ravel(), bb.ravel()]

            ab_T = torch.tensor(ab).float()
            X_clone=torch.clone(ab_T)
            X_clone.requires_grad=True
            z=defense_model(X_clone)
            y_pred=torch.max(z,dim=1).indices
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss=criterion(z,y_pred) #loss at each data point
            loss_sum=torch.sum(loss)
            loss_sum.backward()
            if(att):
                UV=attack_model(ab_T,y_pred).detach().to('cpu').numpy()
                print("attack model plot")
            else:
                UV=X_clone.grad.detach().to('cpu').numpy()
                print("grad plot")
            if(att):
                col="magenta"
            else:
                col="Red"
            U=UV[:,0]
            uu=U.reshape(aa.shape)
            V=UV[:,1]
            vv=V.reshape(aa.shape)
            speed = np.sqrt(uu**2 + vv**2)
            UN = uu/speed
            VN = vv/speed
            axsLeft[i,j].quiver(aa, bb, UN, VN,  # assign to var
                       color=col, 
                       headlength=7)

            axsLeft[i,j].set_xlim([xl+0.1,xu-0.1])
            axsLeft[i,j].set_ylim([yl,yu])
            if j==0:
                axsLeft[i,j].set_ylabel(r'$\delta=$'+str(epsilon))
    axsLeft[0,0].set_title(r'$\lambda^f,\alpha=0$',loc='center')
    axsLeft[0,1].set_title(r'$\lambda^f,\alpha=0.3$',loc='center')
    axsLeft[0,2].set_title(r'$\lambda^f,\alpha=0.6$',loc='center')
    #plt.show()
    plt.savefig('paper_plots/alpha_compare_all.png',dpi=600)
    
alpha_grid_plot()