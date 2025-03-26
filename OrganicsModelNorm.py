# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:00:26 2022

@author: kmccl
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

#from sklearn.decomposition import PCA
from scipy import signal
from scipy import ndimage
from utils import *



# Model
class ORGaNICsCell(nn.Module):
    def __init__(
        self,
        W,
        inputCounts,
        network_params,
        ):
        super().__init__()

        self.weights = W
        self.dt,self.batch_size,_ = inputCounts
        self.params = network_params 
        
        self.device = network_params.device
        self.n_cells = W.n_cells
        
        self.tau_y = network_params.tau_y*self.dt
        self.alpha = network_params.norm.alpha
        self.tau_u = network_params.norm.tau_u*self.dt
        self.tau_w = network_params.norm.tau_w*self.dt
        self.sigma = network_params.norm.sigma
        self.recScale = network_params.norm.recScale
        
        self.noise = network_params.noise
        
        self.Wz = torch.tensor(W.input,dtype=torch.float32,device=self.device,requires_grad=False)
        self.Wy = torch.tensor(W.rec,dtype=torch.float32,device=self.device,requires_grad=False)
        if network_params.Wx_tweak:
            self.Wx = nn.Parameter(torch.tensor(W.shift,dtype=torch.float32,device=self.device,requires_grad=True))
        else:
            self.Wx = torch.tensor(W.shift,dtype=torch.float32,device=self.device,requires_grad=False)
        

    def init_hidden(self, init_params):
        y0 = init_params.y*torch.ones(self.n_cells,1,device=self.device)
        w0 = init_params.w*torch.ones(1,device=self.device)
        u0 = init_params.u*torch.ones(1,device=self.device) 
        
        return y0, w0, u0
        
    def recurrence(
        self, x: Tensor, 
        a_inp: Tensor,
        b_inp: Tensor,
        hidden: Tuple[Tensor, Tensor, Tensor, Tensor],
        normalization: bool=True
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        a = a_inp
        b = b_inp
        
        y, w, u = hidden
        
        # rectify
        #y_plus = y.clamp(0)
        y_plus = y.clamp(0)
        b_plus = b.clamp(0)
        w_plus = w.clamp(0)
        u_plus = u.clamp(max=1)
        
        # compute terms
        z = torch.matmul(self.Wz,x.reshape(x.shape[0],1))
        z = z.reshape((z.size(0),1))
        yy = torch.matmul(self.Wy,y_plus)
        xy = torch.matmul(self.Wx,y_plus)
        
        # compute derivatives, update values
        if normalization:
            
            du = self.dt/self.tau_u * (-u + u * (torch.norm(y_plus)**2 + ((b_plus*self.sigma)/(b_plus + u_plus))**2))
    
            u_new = u + du
            du_plus = u_new.clamp(0) - u_plus

            dw = self.dt/self.tau_w * (-w + w_plus * u_plus + 1 + self.alpha * du_plus/self.dt)
        
            w_new = w + dw
        
            dy = self.dt/self.tau_y * (-y + (b_plus/(1+ b_plus))*z + \
                              (self.recScale/w_plus)*(1/(1+b_plus))*yy) + a*xy # for amplitude of shift to be accurary must be outside taus

        else:
            dy = self.dt/self.tau_y * (-y + (b_plus/(1+ b_plus))*z + \
                              (1/(1+b_plus))*yy) + a*xy
                
            u_new = u
            w_new = w
        
        y_new = y + dy
        
        if self.noise=='additive':
            y_new = y_new + .2*(torch.rand((self.n_cells,1),device=self.device)-.5)
            
        elif self.noise== 'multiplicative': 
            y_new = y_new*1*(torch.rand((self.n_cells,1),device=self.device)+.5)

        return y_new, w_new, u_new
        
    def forward(
        self,
        dir_input: Tensor, 
        a_shift: Tensor,
        b: Tensor,
        hidden: Tuple[Tensor, Tensor, Tensor, Tensor] = None,
        normalization: bool=True
        ):
        
        if hidden is None:
            hidden = self.init_hidden(self.params.init)
        

        output = []
        extras = {'u':[],'w':[]}
        
        for i in range(dir_input.size(0)):
            
            hidden = self.recurrence(dir_input[i],a_shift[i], b[i], hidden,normalization=normalization)
            output.append(hidden[0])
            
            extras['u'].append(hidden[2].item())
            extras['w'].append(hidden[1].item())

        output = torch.stack(output, dim=0).clamp(0.)
        
        for key,val in extras.items():
            extras[key] = np.array(val)
            
        return output, extras

        
class ORGaNICsNet(nn.Module):
    """Recurrent network model."""
    def __init__(
        self,
        inputs,
        weights,
        network_params,
        ):
        super().__init__()

        self.device = network_params.device
        self.inputs = inputs
        self.rnn = ORGaNICsCell(weights,inputs.countParams(),network_params)
        self.params = network_params
        
        self.dt,self.n_timestamps,_ = inputs.countParams()

    def forward(
        self,
        normalization: bool=True
        ) -> Tensor:

        x = torch.as_tensor(self.inputs.dir_one_hot,dtype=torch.float32,device=self.device)
        a_shift = torch.as_tensor(self.inputs.a_shift,dtype=torch.float32,device=self.device)
        b = torch.as_tensor(self.inputs.b_modulator,dtype=torch.float32,device=self.device)
        
        rnn_activity, extraVars = self.rnn(x,a_shift,b,normalization=normalization)
        
        return rnn_activity, extraVars
    
    
class ORGaNICsNetTrain(nn.Module):
    def __init__(
        self,
        inputs,
        weights,
        network_params,
        noise = False,
        ):
        super().__init__()
        
        self.device = network_params.device
        self.inputs = inputs
        self.rnn = ORGaNICsCell(weights,inputs.countParams(),network_params)
        self.params = network_params
        
        self.dt,self.n_timestamps,_ = inputs.countParams()
        
        self.initializeB(network_params.b)
        
        self.smooth_output = network_params.fit.smooth_output
        self.smoothing_kernel_width = network_params.fit.smoothing_kernel_width
    
        
    def initializeB(self,b_params):
        b_start = b_params.b0
        
        if b_params.downSamp:
            n_b_small = int(self.n_timestamps/b_params.downSampRatio)
            self.b_small = nn.Parameter(((b_start/2)*torch.rand(n_b_small,1,device=self.device,requires_grad=True))+(b_start/4))
            self.b_expander = torch.tensor(expanderMat(self.n_timestamps, n_b_small),requires_grad=False,dtype=torch.float32,device=self.device)
            self.b = torch.matmul(self.b_expander,self.b_small)
        else:
            self.b = nn.Parameter(((b_start/2)*torch.rand(self.n_timestamps,1,device=self.device,requires_grad=True))+(b_start/4))
    
    def forward(
        self,
        normalization: bool=True
        ) -> Tensor:
        
        x = torch.as_tensor(self.inputs.dir_one_hot,dtype=torch.float32,device=self.device)
        a_shift = torch.as_tensor(self.inputs.a_shift,dtype=torch.float32,device=self.device)
        
        if self.params.b.downSamp:
            self.b = torch.matmul(self.b_expander,self.b_small)
            
        rnn_activity, _ = self.rnn(x,a_shift,self.b,normalization=normalization)
        
        if self.smooth_output:
            kernel = gaussian_kernel(np.round(self.smoothing_kernel_width/self.dt),device=self.device)
            kernel = kernel.reshape(1,1,-1)
            rnn_activity = rnn_activity.permute(1,2,0)
            rnn_activity = torch.nn.functional.conv1d(rnn_activity,kernel,padding='same')
            rnn_activity = rnn_activity.permute(2,0,1)
        
        return rnn_activity
        
