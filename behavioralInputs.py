# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:45:05 2025

@author: kmcla
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
from utils import *


class inputVariables():
    """set of input variables that can be fed into organics model, doesn't do 
        anything on its own, use subclasses to generte/incorporate inputs"""
    
    def __init__(
            self,
            task_params):
        self.params = task_params
        
        self.dir_one_hot=None
        self.a_shift=None
        self.b_modulator=None
        self.reset_ints=None
        
        self.timestamps=None
        self.dir_vec=None
        self.local_dir_vec=None
        self.distal_speed=None
        self.local_speed=None
        
        # self.timeseries = [
        #     self.dir_one_hot,self.a_shift,self.b_modulator,self.timestamps,
        #     self.dir_vec,self.local_dir_vec,self.distal_speed,self.local_speed
        #     ]     
        
    @property
    def dt(self):
        return self.params.dt
    
    @property
    def n_timestamps(self):
        return self.params.n_timestamps
    
    @property
    def n_dir_increments(self):
        return self.params.n_dir_increments
    
    @property
    def pos_scale(self):
        return self.params.pos_scale
    
    def generateA_shift(self,frame):
        if frame=='local':
            self.a_shift = self.local_speed*self.dt * self.n_dir_increments/self.pos_scale
            
        elif frame=='distal':
            self.a_shift = self.distal_speed*self.dt * self.n_dir_increments/self.pos_scale
        
    def subselectTime(
            self,
            subsel_params,
            ):
        total_time = subsel_params.total_time
        offset_time = subsel_params.offset_time
        
        if total_time is not None:
            batch_size = np.round(total_time*1000/self.dt).astype(int)
            offset_ind = np.round(offset_time*1000/self.dt).astype(int)
            
            if offset_ind+batch_size>self.n_timestamps:
                offset_time=0
                
            self.params.n_timestamps = batch_size
            
            self.dir_one_hot = self.dir_one_hot[offset_ind:(offset_ind+batch_size),:]
            self.a_shift = self.a_shift[offset_ind:(offset_ind+batch_size),:]
            self.timestamps = self.timestamps[:batch_size,:]
            self.dir_vec = self.dir_vec[offset_ind:(offset_ind+batch_size),:]
            self.local_dir_vec = self.local_dir_vec[offset_ind:(offset_ind+batch_size),:]
            self.local_speed = self.local_speed[offset_ind:(offset_ind+batch_size)]
            self.distal_speed = self.distal_speed[offset_ind:(offset_ind+batch_size)]
            
            if self.b_modulator is not None:
                self.b_modulator = self.b_modulator[offset_ind:(offset_ind+batch_size),:]
    
    def countParams(self):
        return [self.dt,self.n_timestamps,self.n_dir_increments]
        
    def inferDir(self):
        dir_inf = np.zeros((self.n_timestamps,1))
        for ii in np.arange(self.reset_ints.shape[0]):
            rel_inds = np.arange(self.reset_ints[ii,0],self.reset_ints[ii,1])
            dir_inf[rel_inds] = self.dir_vec[rel_inds]
            last_dir = self.dir_vec[rel_inds[-1]]
            
            if ii < self.reset_ints.shape[0]-1:
                rel_inds = np.arange(self.reset_ints[ii,1],self.reset_ints[ii+1,0])
            else:
                rel_inds = np.arange(self.reset_ints[ii,1],self.n_timestamps)
            local_snip = self.local_dir_vec[rel_inds] - self.local_dir_vec[rel_inds[0]-1]
            local_snip = local_snip + last_dir
            dir_inf[rel_inds] = local_snip
                
        return dir_inf
    
    def plotInputs(self):
        fig, axs = plt.subplots(5,1)
        
        timeline = self.timestamps/(60*1000)
        axs[0].plot(timeline,self.dir_vec%self.pos_scale, label = "dis dir")
        axs[0].plot(timeline,(self.local_dir_vec+self.dir_vec[0])%self.pos_scale,label = "loc dir")
        axs[0].set_ylabel('Direction (deg)')
        axs[0].legend()
        
        axs[1].plot(timeline,self.distal_speed, label = 'dis speed')
        axs[1].plot(timeline,self.local_speed, label = 'loc speed')
        axs[1].legend()
        [ymin,ymax] = axs[1].get_ylim()
        axs[1].set_ylim(np.min([0,ymin]),np.max([0,ymax]))
        axs[1].sharex(axs[0])
        
        if self.b_modulator is not None:
            axs[2].plot(timeline,self.b_modulator,label = "b mod",color='b')
            axs[2].set_ylabel('b mod')
            axs[2].sharex(axs[0])
        
        axs[3].plot(timeline,self.a_shift,label = "a shift",color='g')
        axs[3].set_ylabel('a mod')
        axs[3].sharex(axs[0])

        x_min = timeline[0,0]
        x_max = timeline[-1,0]
        axs[4].imshow(self.dir_one_hot.T,aspect='auto',interpolation='none',extent=[x_min,x_max,0,self.n_dir_increments])
        axs[4].sharex(axs[0])        
        plt.xlim((x_min,x_max))

        return
    
    def getVariables(self):
        variables = {}
        variables['x_rm'] = self.dir_one_hot
        variables['timeline'] = self.timestamps/1000
        variables['x_rm_vec'] = self.dir_vec
        variables['x_ar_vec'] = self.local_dir_vec
        variables['w'] = self.local_speed*1000
        variables['w_rm'] = self.distal_speed*1000        
        return variables

        
class simulateInputs(inputVariables):
    """produce b modulator through simulation based on preset parameters"""
    
    def __init__(self,
                 task_params):
        super().__init__(task_params)
        
    def generateB(self,bParams):
        b_mod_type = bParams.b_mod_type
        b_scale = bParams.b_scale
        reset_duration = bParams.reset_duration
        n_resets = bParams.n_resets
        b_smooth = bParams.b_smooth
        zeroA = bParams.zeroA
        
        if b_mod_type == 'constant':
            bb = b_scale[0]+np.zeros((self.n_timestamps,1))
        else:
            if b_mod_type == 'random_value':
                bb = np.random.rand((self.n_timestamps,1))
            
            else:
                bb = b_scale[0]+np.zeros((self.n_timestamps,1))
                dur_inds = np.round(reset_duration/self.dt).astype(int)
                if dur_inds <2:
                    dur_inds = 2
                    
                if b_mod_type == 'periodic':
                    reset_inds = np.linspace(0,self.n_timestamps,n_resets+1).astype(int)
                    
                elif b_mod_type == 'rand_interval':
                    reset_inds = np.concatenate(([0],np.random.randint(0, self.n_timestamps, n_resets))).astype(int)
                    
                elif  b_mod_type =='landmark_triggered':
                    reset_inds = -1
                
                for rIdx in reset_inds[:-1]:
                    bb[rIdx:(rIdx+dur_inds)] = b_scale[1]
                    if zeroA:
                        self.a_shift[rIdx:(rIdx+dur_inds)] = 0 
                        
                self.reset_ints = np.column_stack((reset_inds[:-1],reset_inds[:-1]+dur_inds))
                
            if b_smooth:
                sigma = np.round(b_smooth/self.dt)
                filt = np.exp(-(np.arange(-3*sigma, 3*sigma)/sigma)**2/2)
                filt = filt/np.sum(filt)
                bb = ndimage.convolve(bb,filt.reshape(filt.size,1))

        self.b_modulator = bb
        return bb
    
class realInputs(inputVariables):
    """use input data (variables) to create set of input variables"""
    
    def __init__(
            self,
            variables,
            task_params
            ):
        super().__init__(task_params)
        self.timestamps = np.array(variables['timeline'])*1000
        
        self.dir_one_hot = np.array(variables['x_rm'])
        self.dir_vec = np.array(variables['x_rm_vec'])
        self.local_dir_vec = np.array(variables['x_ar_vec'])
        
        self.distal_speed = np.array(variables['w_rm'])/1000
        self.local_speed = np.array(variables['w'])/1000
        
        
        self.params.dt = np.median(np.diff(self.timestamps,axis=0)) #ms
        [self.params.n_timestamps,_] = self.timestamps.shape
        [_,self.params.n_dir_increments] = self.dir_one_hot.shape
        
        self.generateA_shift(task_params.speed.a_shift_frame)
        
        self.subselectTime(self.params.subsel)
        return
        
    
class fullSimInputs(simulateInputs):
    """generate input variables completely through simulation based on preset 
        parameters"""
        
    def __init__(
            self,
            task_params):
        super().__init__(task_params)
        
        self.timestamps = (np.arange(self.n_timestamps).reshape(self.n_timestamps,1))*self.dt
        self.generateSpeed(self.params.speed)
        self.generateA_shift(self.params.speed.a_shift_frame)
        self.generateDirVec()
        self.generateB(self.params.b)
        
        
    def generateSpeed(self,
                      speedParams):
        local_speed_type = speedParams.local_speed_type
        local_speed_scale = speedParams.local_speed_scale
        arena_rotation_speed = speedParams.arena_rotation_speed
        
        if local_speed_type == 'constant':
            speed_local = np.ones((self.n_timestamps,1))*(self.pos_scale*(local_speed_scale)/(60*1000))
                
        elif local_speed_type == 'random': # very slow for some reason
            speed1 = np.random.rand(self.n_timestamps)-.5
            sigma1 = np.round(20*1000/self.dt)
            f1 = np.exp(-(np.arange(-3*sigma1, 3*sigma1)/sigma1)**2/2)
            f1 = f1/np.sum(f1)
            
            sigma2 = np.round(50*1000/self.dt)
            f2 = np.exp(-(np.arange(-3*sigma2, 3*sigma2)/sigma2)**2/2)
            f2 = f2/np.sum(f2)
            
            speed2 = ndimage.convolve(speed1,f1)
            speed3 = ndimage.convolve(speed1,f2)
        
            speed_local = self.params.speed.local_speed_scale *(0.4*speed2 + 0.2*speed3 +.05)
            
        elif local_speed_type == 'random2':
            
            noise1 = np.random.normal(loc=0.0, scale=10, size=self.n_timestamps)
            noise2 = np.random.normal(loc=0, scale=10, size=self.n_timestamps)
            
            winWidth1 = int(np.round(20*1000/self.dt))
            window1 = np.ones(winWidth1) / winWidth1
            noise1 = np.convolve(noise1,window1,mode='same')
            
            winWidth2 = int(np.round(2*1000/self.dt))
            window2 =  np.ones(winWidth2) / winWidth2
            noise2 = np.convolve(noise2,window2,mode='same')
            
            smoothed_signal = np.convolve(noise1, window2, mode='same') + \
                np.convolve(noise2, window2, mode='same')
                
            smoothed_signal = np.convolve(smoothed_signal,window2,mode='same')
            
            offsetVec = np.zeros(smoothed_signal.shape)+.01
            
            speed_local = local_speed_scale*smoothed_signal+offsetVec
            
            
        speed_distal = speed_local + (self.pos_scale*arena_rotation_speed/(60*1000))
        
        self.local_speed = speed_local
        self.distal_speed = speed_distal
        return
    
    def generateDirVec(self):
        local_dir = self.dt*np.cumsum(self.local_speed)
        distal_dir = self.dt*np.cumsum(self.distal_speed)
        
        dir_vec_bin = binVariable(distal_dir%self.pos_scale,self.n_dir_increments,np.array([0,self.pos_scale]))
        
        dir_onehot = np.zeros((self.n_timestamps,self.n_dir_increments))
        dir_onehot[np.arange(self.n_timestamps),dir_vec_bin.astype(int)] = 1
        
        local_dir = local_dir.reshape(self.n_timestamps,1)
        distal_dir = distal_dir.reshape(self.n_timestamps,1)
        self.local_dir_vec = local_dir
        self.dir_vec = distal_dir
        self.dir_one_hot = dir_onehot
        return
    
        
class partSimInputs(simulateInputs,realInputs):
    """Use input data to create a set of input variables, but create b modulator
        through simulation with preset parameters"""
        
    def __init__(
            self,
            variables,
            task_params):
        realInputs.__init__(self,variables,task_params)
        self.generateB(self.params.b)
        