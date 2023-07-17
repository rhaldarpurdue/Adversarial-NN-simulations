# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 05:05:58 2022

@author: rajde
"""
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston,load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

import logging
import time

import os.path

from net import *

from sklearn import datasets

def load_dataset(dataset, input_path, label_path):
    
    if dataset == 'polynomial':
        
        if not os.path.isfile(input_path): 
            X=np.random.uniform(low=-1,high=1,size=2500*2)
            X.resize(2500,2)
            # plt.plot(X[1:100,0],X[1:100,1],'o')

            #eta=2*np.sin(2*np.pi*X[:,1])-X[:,0]**2
            eta= 5*X[:,1]**3-3*X[:,0]**2-2*X[:,1]**2
            prob=1/(1+np.exp(-eta))

            #plt.hist(prob,bins=20)
            #Y=np.random.binomial(1, p=prob)
            #Y=1*(prob>0.3)
            X=X[(prob>0.4) | (prob<0.3),]
            prob=prob[(prob>0.4) | (prob<0.3)]
            Y=1*(prob>0.3)
            #plt.hist(Y,bins=20)
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)

        X=np.load(input_path)
        Y=np.load(label_path)
        
        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X[0:1600,]),torch.from_numpy(Y[0:1600]))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X[1600:len(Y),]),torch.from_numpy(Y[1600:len(Y)]))
    elif dataset == 'moon':
        if not os.path.isfile(input_path): 
            X,Y=datasets.make_moons(n_samples=2500,noise=0.1)
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)
        
        X=np.load(input_path)
        Y=np.load(label_path)
        
        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X[0:2000,]),torch.from_numpy(Y[0:2000]))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X[2000:len(Y),]),torch.from_numpy(Y[2000:len(Y)]))

    elif dataset == 'moon2':
        if not os.path.isfile(input_path): 
            X,Y=datasets.make_moons(n_samples=2500,noise=0.1)
            X[Y==0,1]=X[Y==0,1]+1
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)
        
        X=np.load(input_path)
        Y=np.load(label_path)
        
        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X[0:2000,]),torch.from_numpy(Y[0:2000]))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X[2000:len(Y),]),torch.from_numpy(Y[2000:len(Y)]))    

    elif dataset == 'circles':
        if not os.path.isfile(input_path): 
            X,Y=datasets.make_circles(n_samples=2500,noise=0.1, factor= 0.5)
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)
        
        X=np.load(input_path)
        Y=np.load(label_path)
        
        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X[0:2000,]),torch.from_numpy(Y[0:2000]))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X[2000:len(Y),]),torch.from_numpy(Y[2000:len(Y)]))
        
    elif dataset == 'blobs':
        if not os.path.isfile(input_path): 
            X,Y= datasets.make_blobs(n_samples=2500, random_state=8)
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)
        
        X=np.load(input_path)
        Y=np.load(label_path)
        
        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X[0:2000,]),torch.from_numpy(Y[0:2000]))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X[2000:len(Y),]),torch.from_numpy(Y[2000:len(Y)]))
    elif dataset == 'streaks':
        if not os.path.isfile(input_path): 
            random_state = 170
            X, Y = datasets.make_blobs(n_samples=2500, random_state=random_state)
            transformation = [[0.6, -0.6], [-0.4, 0.8]]
            X = np.dot(X, transformation)
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)
        
        X=np.load(input_path)
        Y=np.load(label_path)
        
        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X[0:2000,]),torch.from_numpy(Y[0:2000]))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X[2000:len(Y),]),torch.from_numpy(Y[2000:len(Y)]))
    elif dataset== 'boston':
        if not os.path.isfile(input_path):
            bos = load_boston() # boston dataset it will be removed though in future versions from scikit
            df = pd.DataFrame(bos.data)
            df.columns = bos.feature_names
            df['Price'] = bos.target
            df.head()

            data = df[df.columns[:-1]]
            data = data.apply(
                lambda x: (x - x.mean()) / x.std()
            )

            data['Price'] = df.Price

            X = data.drop('Price', axis=1).to_numpy()
            Y = data['Price'].to_numpy()
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)
        X=np.load(input_path)
        Y=np.load(label_path)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X_train),torch.from_numpy(Y_train))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X_test),torch.from_numpy(Y_test))
    elif dataset== 'diabetes':
        if not os.path.isfile(input_path):
            bos = load_diabetes() # boston dataset it will be removed though in future versions from scikit
            df = pd.DataFrame(bos.data)
            df.columns = bos.feature_names
            df['Disease'] = bos.target
            df.head()

            data = df[df.columns[:-1]]
            data = data.apply(
                lambda x: (x - x.mean()) / x.std()
            )

            data['Disease'] = df.Disease

            X = data.drop('Disease', axis=1).to_numpy()
            Y = data['Disease'].to_numpy()
            np.save(input_path,X,allow_pickle=False)
            np.save(label_path,Y,allow_pickle=False)
        X=np.load(input_path)
        Y=np.load(label_path)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_data=torch.utils.data.TensorDataset(torch.from_numpy(X_train),torch.from_numpy(Y_train))
        test_data=torch.utils.data.TensorDataset(torch.from_numpy(X_test),torch.from_numpy(Y_test))
        
    return X, Y, train_data, test_data
    