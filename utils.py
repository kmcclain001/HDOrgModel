# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:17:04 2025

@author: kmcla
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage

def binVariable(x,nBins,rangeX):
    
    xScaled = (x-rangeX[0])*nBins/(rangeX[1]-rangeX[0])
    out = xScaled.astype(int)
    
    return out

def repZero(x):
    if x==0:
        return 1
    else:
        return x
    
def gaus_mat(
        n_row:int=10,
        n_col:int=10,
        width:float=.2,
        circular:bool=True
        ):
    
    tuning_width = int(width*n_col)
    mid_int = int(tuning_width/2)
    
    raised_cos = .5*np.cos(np.linspace(-np.pi,np.pi,tuning_width))+.5
    if circular:
        peak_inds = np.linspace(0, n_col, n_row+1).astype(int)
    else:
        peak_inds = np.linspace(0, n_col-1, n_row).astype(int)
        
    M = np.zeros((n_row,n_col))
    for rIdx in np.arange(n_row):
        start_ind = peak_inds[rIdx]-mid_int
        end_ind = tuning_width+start_ind
        if end_ind < n_col:
            M[rIdx,np.arange(start_ind,(tuning_width+start_ind))] = raised_cos
        else:
            cutoff = n_col-start_ind
            M[rIdx,start_ind:n_col] = raised_cos[:cutoff]
            M[rIdx,0:(raised_cos.size-cutoff)] = raised_cos[cutoff:]
         
    return M

def gaussian_kernel(size, alpha=2.5,for_torch=True,device=None):
    """
    Generate a 1D Gaussian kernel.
    """
    sigma = (size-1)/(2*alpha)
    
    if for_torch:
        coords = torch.arange(size, dtype=torch.float32,device=device)
        coords -= (size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        return g / g.sum()
    else:
        coords = np.arange(size)
        coords -= (size - 1) / 2.0
        g = np.exp(-(coords**2) / (2 * sigma**2))
        return g / g.sum()
    
def expanderMat(size_big,size_small):
    
    mat = np.zeros((size_big,size_small))
    divisions = np.linspace(0, size_big, size_small).astype('int')
    
    for ii in np.arange(divisions.size-1):
        mat[divisions[ii]:divisions[ii+1],ii] = 1
        
    return mat

def cut(x):
    
    return x.detach().cpu().numpy()