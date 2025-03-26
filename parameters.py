# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:13:37 2025

@author: kmcla
"""
import numpy as np
import torch
import pickle
from utils import *


class metaParam:
    def __init__(self,task,weight,network):
        self.task = task
        self.weight = weight
        self.network = network
        
    def save_to_file(self, filename):
        """Serialize the entire MetaParam object using Pickle."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename):
        """Load a MetaParam object from a Pickle file."""
        with open(filename, "rb") as f:
            return pickle.load(f)
        
    
class parameters:
    def __init__(self):
        pass
        
    def update(self, *args):
        
        """Updates attributes using alternating key-value pairs."""
        if len(args) % 2 != 0:
            raise ValueError("Arguments must be in key-value pairs.")
        
        for i in range(0, len(args), 2):
            key, value = args[i], args[i + 1]
            parts = key.split('.')  # Support nested attributes using dot notation
            target = self
       
            # Traverse through nested attributes
            for attr in parts[:-1]:
                if hasattr(target, attr):
                    target = getattr(target, attr)
                else:
                    raise AttributeError(f"'taskParams' object has no attribute '{attr}'")
       
            final_attr = parts[-1]
            if hasattr(target, final_attr):
                setattr(target, final_attr, value)
            else:
                raise AttributeError(f"'{target.__class__.__name__}' object has no attribute '{final_attr}'")
    

class taskParams(parameters):
    
    def __init__(self):
        super().__init__()
        self.dt=10 #ms
        self.n_dir_increments=100
        self.pos_scale=360
        self._total_time = 300 #s
        self._n_timestamps = np.round(self._total_time*1000/self.dt).astype(int)
        self.b = self.bParams(self)
        self.speed = self.speedParams()
        self.subsel = self.subselectParams()
        
    @property
    def n_timestamps(self):
        return self._n_timestamps
    
    @n_timestamps.setter
    def n_timestamps(self, value):
        self._n_timestamps = value
        self._total_time = self._n_timestamps*self.dt/1000
        
    @property 
    def total_time(self):
        return self._total_time
    
    @total_time.setter
    def total_time(self,value):
        self._total_time = value
        self._n_timestamps = np.round(self._total_time*1000/self.dt).astype(int)
    
    class bParams:
        def __init__(self,outer):
            self.b_mod_type='periodic'
            self.b_scale=[0,1]
            self.reset_duration=1000
            self.b_smooth=False
            self.n_resets=np.round(3*outer.total_time/60).astype(int)
            self.zeroA = True
            
    class speedParams:
        def __init__(self):
            self.a_shift_frame='local'
            self.arena_rotation_speed=1
            self.local_speed_type='constant'
            self.local_speed_scale=0.5
            
    class subselectParams:
        """implemented automatically for real and partial inputs, not full sim"""
        def __init__(self):
            self.total_time=None #s must use this one for real inputs
            self.offset_time=10 #s
            
    
class weightParams(parameters):
    
    def __init__(self):
        super().__init__()
        self.n_cells = 25
        self.n_dir_increments = 100
        self.real_inds = np.arange(int(self.n_cells))
        self.ghost = self.ghostParams()
        self.input = self.inputParams()
        self.rec = self.recParams()
        self.shift = self.shiftParams()
    
    class ghostParams:
        def __init__(self):
            self.ghost_cell_insert=None
            self.ghost_inds=None
            self.tuning_width=0.35
        
    class inputParams:
        def __init__(self):
            self.tuning_width=0.25
            
    class recParams:
        def __init__(self):
            self.scale_rec=None # can be scaler
            self.identity_rec=False
            self.k_filt = np.array([[0.02807382, -0.060944743, -0.073386624, 0.41472545, 0.7973934, 0.41472545, -0.073386624, -0.060944743, 0.02807382]])

    class shiftParams:
        def __init__(self):
            self.d_filt = np.array([[-0.10689, -0.28461, 0.0,  0.28461,  0.10689]])
            self.inv_alpha = 0.2
            
            
class networkParams(parameters):
    
    def __init__(self, task_params=None):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  
        else:
            self.device = torch.device("cpu")
            
        self.tau_y = 10 # 2-3 dt
        self.Wx_tweak = False
        self.noise = None # "additive" or "multiplicative"
        self.norm = self.normParams()
        self.init = self.initialParams()
        self.b = self.bParams()
        self.fit = self.fitParams()
        
    class normParams:
        def __init__(self):
            self.simulation = True
            self.alpha = 1000
            self.tau_u = 15 # 3-4 dt
            self.tau_w = 5 # <=tau_y, >dt
            self.sigma = 0.5
            self.recScale = 1
            
    class initialParams:
        def __init__(self):
            self.y = 0.1
            self.w = 1
            self.u = 0.1
            
    class bParams:
        def __init__(self):
            self.b0 = 0.1
            self.downSamp = False
            self.downSampRatio = 100
            
    class fitParams:
        def __init__(self):
            self.loss = 'MSE'
            self.optim ='adam'
            self.learningRate = 0.002
            self.n_training_steps = [5,0]
            self.b_scale = [0, 10]
            self.print_steps = True
            self.smooth_output = False
            self.smoothing_kernel_width = 20
            self.bRegularize = None # or 'L1', 'L2',
            self.bRegScale = 1

