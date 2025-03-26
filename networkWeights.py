# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:52:10 2025

@author: kmcla
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
from utils import *


class weights:
    
    def __init__(self,weight_params):
        
        self.params = weight_params
        
        self.input = None
        self.rec = None
        self.shift = None
        
    @property
    def n_cells(self):
        return self.params.n_cells
    
    @property
    def n_dir_increments(self):
        return self.params.n_dir_increments
    
    @property
    def real_inds(self):
        return self.params.real_inds
    
    @property
    def ghost_inds(self):
        return self.params.ghost.ghost_inds
    
    def generateWy(self,rec_params):
        
        identity_rec = rec_params.identity_rec
        k = rec_params.k_filt
        scale_rec = rec_params.scale_rec
        
        if identity_rec:
            self.rec = np.eye(self.n_cells)
            return
        
        R = np.eye(self.n_cells) #OG basis
            
        R = signal.convolve2d(R,k,mode='same',boundary='wrap')
        u,s,vh = np.linalg.svd(R)
        s[s>=1] = 1
        smat = np.diag(s)
        R = np.dot(vh.T,np.dot(smat,vh))
    
        self.rec = R
        if scale_rec is not None:
            self.scaleRec(scale_rec)
            
        return
    
    def scaleRec(self,scaler):
        self.rec = self.rec*scaleRec
        return
    
    def generateWx(self,shift_params):
        
        dfilter = shift_params.d_filt
        alpha = shift_params.inv_alpha
        
        D = signal.convolve2d(np.eye(self.n_dir_increments), dfilter.T,mode='same',boundary='wrap')
       
        y = self.input
        U,s,Vh = np.linalg.svd(y,full_matrices=False)

        r_s = s/(s**2+alpha)
        y_inv = (Vh.T)@(np.expand_dims(r_s,axis=1)*U.T)
        
        self.shift = y@D@y_inv
        return
    
    def addGhostCells(self,ghost_params):
        
        ghostCellInsert = ghost_params.ghost_cell_insert
        tuningWidth = ghost_params.tuning_width
        
        nGhost = np.size(ghostCellInsert)
        nDirInc = self.n_dir_increments
        
        #find peaks of real cell tuning curves
        peak_loc = np.argmax(self.input,1)
        peak_loc = np.concatenate(([0],peak_loc,[nDirInc]))
        
        #compute peaks of ghost cells between real cells
        unique_inserts, unique_ind = np.unique(ghostCellInsert, return_inverse=True)
        
        ghostPeaks = np.zeros(nGhost)
        for a in np.arange(np.size(unique_inserts)):
            thisInsert = unique_inserts[a]
            nGhosts = np.sum(unique_ind==a)
            newPeaks = np.round(np.linspace(peak_loc[thisInsert],peak_loc[thisInsert+1],nGhosts+2))
            ghostPeaks[unique_ind==a] = newPeaks[1:-1]
            
        megaWz = gaus_mat(nDirInc,nDirInc,width=tuningWidth)
        
        # find indices of ghost cells and real cells
        self.params.ghost.ghost_inds = np.cumsum(np.concatenate(([ghostCellInsert[0]],np.diff(ghostCellInsert)+1))).astype('int')
        self.params.real_inds = np.argwhere(~(np.isin(np.arange(0,nGhost+self.n_cells),self.ghost_inds))).reshape([-1]).astype('int')
        
        # replace weights mats with new weights including ghosts
        self.params.n_cells = nGhost+self.n_cells
        
        Wz = np.zeros([self.n_cells,nDirInc])
        Wz[self.real_inds,:] = self.input
        Wz[self.ghost_inds,:] = megaWz[ghostPeaks.astype('int'),:]
        self.input = Wz
            
        self.generateWy(self.params.rec)
        self.generateWx(self.params.shift)
        return
    
    def plotWeights(self):
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(self.input,aspect='auto',interpolation='none')
        axs[0].set(title="Wz, input")
        axs[1].imshow(self.rec,interpolation='none')
        axs[1].set(title="Wy, recurrent static")
        axs[2].imshow(self.shift,interpolation='none')
        axs[2].set(title="Wx, recurrent shift")
        
class fakeWeights(weights):
    
    def __init__(self,weight_params):
        super().__init__(weight_params)
        
        self.generateWz(self.params.input)
        self.generateWy(self.params.rec)
        self.generateWx(self.params.shift)
        
        if self.params.ghost.ghost_cell_insert is not None:
            self.addGhostCells(self.params.ghost)
            
        
    def generateWz(self,input_params):
        tuning_width = input_params.tuning_width
        
        self.input = gaus_mat(self.n_cells,self.n_dir_increments,tuning_width)
        return
    

class realWeights(weights):
    
    def __init__(self,weightMats,weight_params):
        super().__init__(weight_params)
        
        self.input = np.array(weightMats['Wxy'])
        self.shift = np.array(weightMats['Wsy'])
        self.params.n_cells = self.shift.shape[0]
        
        if self.params.rec.identity_rec:
            self.generateWy(self.params.rec)
        else:
            self.rec = np.array(weightMats['Wyy'])
            
        if weight_params.ghost.ghost_cell_insert is not None:
            self.addGhostCells(weight_params.ghost)
        
        
        