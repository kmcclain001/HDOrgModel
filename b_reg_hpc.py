#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:56:33 2025

@author: kathrynmcclain
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from collections import deque
from typing import Tuple
from functools import partial
from sklearn.decomposition import PCA
from scipy import signal
from scipy import ndimage
from scipy.io import loadmat
from scipy.stats import zscore
import pickle
from datetime import datetime
import os

from utils import *
from parameters import *
from behavioralInputs import *
from networkWeights import *
from HDexperiment import *

# choose device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
# set up output folder

#savePath = '/scratch/km3911/pytorch/use_20250305/outputData/'
savePath = 'C:\\Users\\kmcla\\Dropbox\\headDirection\\current_code\\outputData'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
dataFolder = os.path.join(savePath, f"exp_output_{timestamp}")
os.makedirs(dataFolder, exist_ok=True)

# Load data

inputDataPath = 'input_data/'
sessionIdx = '14'
varFile = inputDataPath+'variables_session_'+sessionIdx+'.mat'
weightFile = inputDataPath+'weights_session_'+sessionIdx+'.mat'

variables = loadmat(varFile)
weightMat = loadmat(weightFile)

# Task

# set task parameters
task_param = taskParams()
task_param.update('total_time',120,
                  'b.n_resets',3,
                  'b.zeroA',True,
                  'b.reset_duration',1000)
inputs = partSimInputs(variables,task_param)
#inputs = fullSimInputs(task_param)

# Weights

# set weight parameters
weight_param = weightParams()
weight_param.update('shift.inv_alpha',2)
#weights = realWeights(weightMat,weight_param)
weights = fakeWeights(weight_param)

# Network

# set network parameters
net_param = networkParams()
net_param.update('norm.recScale',1.0001,
                  'fit.n_training_steps',[0,100],
                  'fit.learningRate',0.01)

for ii in np.array([0,0.001,0.01,0.05,0.1,1,5]):
    #for ii in np.array([0.1]):
    
    net_param.update('fit.bRegularize','L1',
                     'fit.bRegScale',ii)
    
    # Simulate 
    
    trueActivity,_ = simulate(inputs,weights,net_param)
    
    # Fit
    
    fitting = modelFit(inputs,weights,net_param)
    results = fitting.run(trueActivity)
    
    # Save results to a file
    filename = os.path.join(dataFolder, f"lambda_{ii}.pkl")
    
    with open(filename, "wb") as f:
        pickle.dump(results, f)
        
