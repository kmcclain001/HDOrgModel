#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:07:35 2025

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

import sys
import argparse
    
def main(paramFile,dataDir,outputDir):
    
    # load parameters
    paramPack = metaParam.load_from_file(paramFile)
    task_param = paramPack.task
    weight_param = paramPack.weight
    net_param = paramPack.network

    
    # Load data
    varFile = dataDir+'variables.mat'
    weightFile = dataDir+'weights.mat'
    
    variables = loadmat(varFile)
    weightMat = loadmat(weightFile)
    
    activity_true = np.array(variables['frReal'])
    
    # Task
    inputs = realInputs(variables,task_param)
    
    # Weights
    weights = realWeights(weightMat,weight_param)
    
    # real activity
    activity_true = prepRealY(activity_true,task_param,net_param)
    
    # Run fitting
    fitting = modelFit(inputs,weights,net_param)
    results = fitting.run(activity_true)
    
    #CLUSTER
    # Save results to a file
    filename = os.path.join(outputDir, f"realFitResults.pkl")
    
    with open(filename, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args.config_path,args.input_path,args.output_path)