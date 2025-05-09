import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing

import pandas as pd
import os
from multi_gaston import data_processing, multi_gaston

# =========================Data Processing====================================================
# Load the metabolite list containing metabolite names, mass/s and formula
metabolites = pd.read_excel('livm_meta.xlsx',header=None).to_numpy()
metabolites = metabolites[1:,:]
# Load and process the raw data, the TIC is plotted to visualize the sample
X,Y,S,A = data_processing.process('m163-165_neg_02_cluster_ep_n4_tgt.csv', plot = True, save_dir = 'm163-165_neg_02_cluster_ep_n4.png')
np.savetxt('m163-165_neg_02_cluster_ep_n4_processed.csv',A, delimiter=",")

# scale and convert the data into pytorch tensors
scaler = preprocessing.StandardScaler().fit(A)
A_scaled = scaler.transform(A)

scaler = preprocessing.StandardScaler().fit(S)
S_scaled = scaler.transform(S)

S_torch=torch.tensor(S_scaled,dtype=torch.float32)
A_torch=torch.tensor(A_scaled,dtype=torch.float32)


# =========================Neural Net parameters: can be changed==============================
# NN architectures are encoded as list, eg [20,20] means two hidden layers of size 20 hidden neurons
isodepth_arch=[500] # architecture for isodepth neural network d(x,y) : R^2 -> R 

num_epochs = 10000 # number of epochs to train NN (NOTE: it is sometimes beneficial to train longer)
checkpoint = 1000 # save model after number of epochs = multiple of checkpoint
out_dir='m163-165_neg_02_cluster_ep_n4' # folder to save model runs
optimizer = "adam"
num_restarts=10
K = 2
lasso_coefficient = 0.001

seed_list=range(num_restarts)
for seed in seed_list:
    print(f'training neural network for seed {seed}')
    out_dir_seed=f"{out_dir}/rep{seed}"
    os.makedirs(out_dir_seed, exist_ok=True)
    mod, loss_list = multi_gaston.train(S_torch, A_torch,
                          S_hidden_list=isodepth_arch, A_hidden_list=[], 
                          epochs=num_epochs, checkpoint=checkpoint, 
                          A_linear = False, lasso_lambda = lasso_coefficient, K = K,
                          save_dir=out_dir_seed, optim=optimizer, seed=seed, save_final=True)