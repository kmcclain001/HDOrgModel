# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:13:27 2025

@author: kmcla
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from collections import deque
from typing import Tuple
from functools import partial
from sklearn.decomposition import PCA
from scipy import signal
from scipy import ndimage
from scipy.io import loadmat
from scipy.stats import zscore
import matplotlib.gridspec as gridspec
from utils import *
from OrganicsModelNorm import *
        
        
class modelFitOutput():

    def __init__(
            self,
            fit_params,
            n_timestamps,
            ):
        self.n_timestamps = n_timestamps
        self.n_training_steps = fit_params.n_training_steps
        self.b_cache = np.zeros((np.sum(self.n_training_steps),self.n_timestamps))
        self.loss_cache = np.zeros(np.sum(self.n_training_steps))
        self.mean_error = np.zeros((np.sum(self.n_training_steps),self.n_timestamps))
        self.ii = 0
        
    def addStep(self,b,l,e,):

        self.b_cache[self.ii,:] = b
        self.loss_cache[self.ii] = l
        self.mean_error[self.ii,:] = e
        self.ii+=1
        
    def wrapUp(
            self,
            timestamps,
            activityTrue,
            activityEnd,
            b_true=None,
            ):
        self.timestamps = timestamps
        self.activityTrue = activityTrue
        self.activityEnd = activityEnd
        self.b_true = b_true
        
    def plotZ(self):
        
        timeline = self.timestamps/(60*1000)
        
        nPlot = 8
        fig, axs = plt.subplots(nPlot,1,layout="constrained")
        
        x_min = np.min(timeline)
        x_max = np.max(timeline)
        
        ii = 0
        im = axs[ii].imshow(self.b_cache,aspect='auto',interpolation='none',extent=[x_min,x_max,self.n_training_steps,0])
        axs[ii].set_ylabel('b training')
        
        ii+=1
        axs[ii] = plt.subplot(nPlot,1,ii+1)
        axs[ii].scatter(timeline,self.b_cache[0,:],s=0.2)
        axs[ii].set_ylabel('b(i=0)')
        axs[ii].sharex(axs[0])
        
        ii+=1
        axs[ii] = plt.subplot(nPlot,1,ii+1)
        axs[ii].scatter(timeline,self.b_cache[-1,:],s=0.2)
        axs[ii].set_ylabel('b(i=end)')
        axs[ii].sharex(axs[0])
        
        ii+=1
        axs[ii] = plt.subplot(nPlot,1,ii+1)
        axs[ii].plot(timeline,self.activityTrue[:,:,0])
        axs[ii].set_ylabel('y true')
        axs[ii].sharex(axs[0])
        plt.xlim((0,max(timeline)))
        y2_min, y2_max = plt.ylim()
        
        ii+=1
        axs[ii] = plt.subplot(nPlot,1,ii+1)
        axs[ii].plot(timeline,self.activityEnd[:,[ 2,  6,  9, 11, 13],0]) # TEMPORARY !!!
        axs[ii].set_ylabel('y(i=end)')
        y1_min, y1_max = plt.ylim()
        axs[ii].sharex(axs[0])
        
        y_min = np.min([y1_min,y2_min])
        y_max = np.max([y1_max,y2_max])
        
        axs[ii-1].set_ylim((y_min,y_max))
        axs[ii].set_ylim((y_min,y_max))
        
        ii+=1
        im = axs[ii].imshow(self.mean_error,aspect='auto',interpolation='none',extent=[x_min,x_max,self.n_training_steps,0])
        axs[ii].set_ylabel('error(t,i)')
        axs[ii].set_xlabel('timestep = t')
        
        #axs[ii].sharex(axs[0])
        
        ii+=1
        axs[ii].plot(timeline,self.loss_cache)
        axs[ii].set_ylabel('error_y(i)')
        b_error = np.linalg.norm(self.b_cache-self.b_true.T,axis=1)
        
        ii+=1
        axs[ii].plot(self.b_error)
        axs[ii].set_ylabel('error_b(y))')
        axs[ii].set_xlabel('training iteration = i')
        
    def plotA(self):
        
        timeline = self.timestamps/(60*1000)
        n_iter = np.sum(self.n_training_steps)
        
        fig = plt.figure(figsize=(8, 14))
        gs = gridspec.GridSpec(8, 3, height_ratios=[1, 0.5, 0.5, 1, 1, 1,0.1,1], width_ratios=[1,.1,1])  
        
        x_min = np.min(timeline)
        x_max = np.max(timeline)
        
        # b training
        ax1 = plt.subplot(gs[0, :])  
        ax1.imshow(self.b_cache,aspect='auto',interpolation='none',extent=[x_min,x_max,n_iter,0])
        ax1.set_ylabel('b training')
        
        # initial b
        ax2 = plt.subplot(gs[1, :])  
        ax2.scatter(timeline,self.b_cache[0,:],s=0.2)
        ax2.set_ylabel('b(i=0)')
        ax2.sharex(ax1)
                
        # final b
        ax3 = plt.subplot(gs[2, :])  
        ax3.scatter(timeline,self.b_cache[-1,:],s=0.2)
        ax3.set_ylabel('b(i=end)')
        ax3.sharex(ax1)
        
        # true activity
        ax4 = plt.subplot(gs[3,:]) 
        ax4.plot(timeline,self.activityTrue[:,:,0])
        ax4.set_ylabel('y true')
        ax4.sharex(ax1)
        plt.xlim((0,max(timeline)))
        y2_min, y2_max = plt.ylim()
        
        # final learned activity
        ax5 = plt.subplot(gs[4, :]) 
        ax5.plot(timeline,self.activityEnd[:,[ 2,  6,  9, 11, 13],0]) # TEMPORARY !!!
        ax5.set_ylabel('y(i=end)')
        y1_min, y1_max = plt.ylim()
        ax5.sharex(ax1)
        
        y_min = np.min([y1_min,y2_min])
        y_max = np.max([y1_max,y2_max])
        
        ax4.set_ylim((y_min,y_max))
        ax5.set_ylim((y_min,y_max))
        
        # mean error over time and learning
        ax6 = plt.subplot(gs[5, :]) 
        ax6.imshow(self.mean_error,aspect='auto',interpolation='none',extent=[x_min,x_max,n_iter,0])
        ax6.set_ylabel('error(t,i)')
        ax6.set_xlabel('timestep = t')
        
        # change in loss y
        ax7 = plt.subplot(gs[7, 0]) 
        ax7.plot(self.loss_cache)
        ax7.set_ylabel('error_y(i)')
        
        # change in loss b
        ax8 = plt.subplot(gs[7, 2]) 
        b_error = np.linalg.norm(self.b_cache-self.b_true.T,axis=1)
        ax8.plot(b_error)
        ax8.set_ylabel('error_b(y))')
        ax8.set_xlabel('training iteration = i')
        
class modelFit():
    
    def __init__(
            self,
            inputs,
            weights,
            net_params,
            ):
        self.params = net_params
        self.dt,self.n_timestamps,_ = inputs.countParams()
        self.net_train = ORGaNICsNetTrain(inputs, weights, net_params)

        self.inputs = inputs
        self.weights = weights
        
    def run(
            self,
            trueActivity
            ):
        n_training_steps = self.params.fit.n_training_steps
        lr = self.params.fit.learningRate
        lossType = self.params.fit.loss
        optimType = self.params.fit.optim
        b_scale = self.params.fit.b_scale
        b_reg = self.params.fit.bRegularize
        b_reg_scale = self.params.fit.bRegScale
        
        fullTrueActivity = cut(trueActivity)
        if trueActivity.shape[1] == self.weights.n_cells:
            fullTmpInds = np.arange(0,self.weights.n_cells)
            trueActivity = trueActivity[:,self.weights.real_inds,:]
        elif trueActivity.shape[1] == np.size(self.weights.real_inds):
            fullTmpInds = self.weights.real_inds
            
        else: # if true activity doesn't match total number or real number of cells
            raise ValueError("number of cells in true activity doesn't make sense")

            
        if lossType == 'MSE':
            criterion = nn.MSELoss()
        else: 
            pass
            
        if optimType =='adam':
            optimizer = optim.Adam(self.net_train.parameters(),lr=lr)
        else:
           pass
            
        #scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.75,patience=2) # make larger patience
        norm_condition = np.ones(np.sum(n_training_steps))
        norm_condition[0:n_training_steps[0]]=0
        
        output = modelFitOutput(self.params.fit,self.n_timestamps)
        
        for i in range(np.sum(n_training_steps)):
            if self.params.fit.print_steps:
                print(str(i))
                
            b_old = cut(self.net_train.b)[:,0]
            activity_temp = self.net_train(normalization=norm_condition[i])    
            loss_term = criterion(activity_temp[:,self.weights.real_inds,:],trueActivity)
              
            if b_reg == 'L1':
                reg_term = torch.norm(self.net_train.b,p=1)/(self.n_timestamps)
                loss = loss_term+b_reg_scale*reg_term
            elif b_reg == 'L2':
                reg_term = torch.norm(self.net_train.b,p=2)/(np.sqrt(self.n_timestamps))
                loss = loss_term + b_reg_scale*reg_term
            else:
                loss = loss_term
                
            loss.backward()
            optimizer.step()
            
            #scheduler.step(loss)
            # current_lr = optimizer.param_groups[0]['lr']
            # print(f"Epoch {i}: Learning Rate = {current_lr:.6f}")
            print(activity_temp.device)
            
            with torch.no_grad():
                self.net_train.b.clamp_(b_scale[0],b_scale[1])
                
            meanErr = np.sqrt(np.mean(np.square(cut(activity_temp)[:,fullTmpInds,0]-fullTrueActivity[:,:,0]),axis=1))
            output.addStep(b_old,cut(loss),meanErr)
        
        output.wrapUp(self.inputs.timestamps,cut(trueActivity),cut(activity_temp),self.inputs.b_modulator)
        
        return output
    

def simulate(inputs,weights,net_params):
    
    net_gen = ORGaNICsNet(inputs,weights,net_params)
    with torch.no_grad():
        activity_true,extras = net_gen(normalization=net_params.norm.simulation)
        
    return activity_true.detach().clone(),extras
        
def plotActivity(inputs,activity,cutNan=True):
            
    timeline = inputs.timestamps/(60*1000)
    
    if torch.is_tensor(activity):
        activity = cut(activity)
    if len(activity.shape) == 3:
        activity = activity[:,:,0]
            
    if cutNan:
        nan_inds = np.argwhere(~np.isnan(np.sum(activity,1)))[:,0]
        if nan_inds.any():
            activity = activity[nan_inds,:]
            timeline = timeline[nan_inds]
        
    #simulated activity
    fig,axs = plt.subplots(2,1)
    axs[0].imshow(activity.T,aspect='auto',interpolation='none',extent=[timeline[0,0],timeline[-1,0],0,activity.shape[1]])
    
    axs[1] = plt.subplot(2,1,2)
    axs[1].plot(timeline,activity)
    axs[1].set_ylabel('activity true')
    axs[1].sharex(axs[0])
    plt.xlim((0,timeline[-1]))

        
def plotExtras(inputs,extras,cutNan=True):
    
    timeline = inputs.timestamps/(60*1000)
    
    if cutNan:
        nan_inds = np.argwhere(~np.isnan(list(extras.values())[0]))[:,0]
        if nan_inds.any():
            timeline = timeline[nan_inds]
            for key,val in extras.items():
                extras[key] = val[nan_inds]
                
    fig,axs = plt.subplots(2,1)
    axs[0].plot(timeline,extras['u'])
    axs[0].set_ylabel('u(t)')
    
    axs[1].plot(timeline,extras['w'])
    axs[1].set_ylabel('w(t)')
    
def prepRealY(activity,task_params,net_params):
    
    #reshape
    if len(activity.shape)==2:
        activity = activity.reshape(activity.shape[0],activity.shape[1],1)
        
    #subselect time
    if task_params.subsel.total_time is not None:
        offset_ind = np.round(task_params.subsel.offset_time*1000/task_params.dt).astype(int)
        activity = activity[offset_ind:(offset_ind+task_params.n_timestamps),:,:]
    
    activity = torch.as_tensor(activity,dtype=torch.float32,device=net_params.device)
    return activity
    
    
    