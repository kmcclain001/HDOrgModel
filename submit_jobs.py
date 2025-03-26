#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:08:49 2025

@author: kathrynmcclain
"""

import os
import copy
import configparser
from itertools import product
from copy import deepcopy
import subprocess
import time
import shutil
from datetime import datetime
import pickle

from parameters import *


def create_sbatch(job_name,template,param,inpDir,outDir):
    """Reads the SBATCH template, fills in values, and writes a new SBATCH file."""
   # sbatch_path = os.path.join(outDir, f"{job_name}.sbatch")
    sbatch_path = f"{job_name}.SBATCH"

    with open(template, "r") as template_file:
        sbatch_content = template_file.read()

    # Replace placeholders with actual values
    sbatch_content = sbatch_content.format(jobName=job_name,paramFile=param,dataDir=inpDir,outputDir=outDir)

    with open(sbatch_path, "w") as sbatch_file:
        sbatch_file.write(sbatch_content)

    return sbatch_path  # Return the path to the generated SBATCH file
    
def main():
   
    # Input data path
    inputPath = 'input_data/'
    
    # sbatch info
    template_file = "sb_template.sbatch"
    
    # Output path
        # folder for each run
    #savePath = '/scratch/km3911/pytorch/use_20250324/outputData/'
    savePath = '/Users/kathrynmcclain/Dropbox/headDirection/extraFolder/outputData'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runDir = os.path.join(savePath, f"exp_output_{timestamp}")
    os.makedirs(runDir, exist_ok=True)

    # Define default paramaters
        # Task
    task_param = taskParams()
    task_param.update('subsel.total_time',1*60)

        # Weights
    weight_param = weightParams()
    weight_param.update('shift.inv_alpha',2,
                        'ghost.ghost_cell_insert',[0,0,1,1,1,2,2,3,4,5,5,5,5,5,5,5,5,5,5])

        # Network
    net_param = networkParams()
    net_param.update('norm.recScale',1.0001,
                      'fit.n_training_steps',[0,10])
    net_param.update('tau_y',20,
                     'norm.tau_u',30)

    # Iterate over changing parameter
    learningRates = [0.0001,0.001,0.01,0.05]
    count = 1
    for lr in learningRates:
        
        net_param.update('fit.learningRate',lr)
        
        # save paramters
        iterDir = os.path.join(runDir,f"LR_{lr}")
        os.makedirs(iterDir, exist_ok=True)
        paramPack = metaParam(task_param,weight_param,net_param)
        p_file = os.path.join(iterDir,f"config_LR_{lr}.pkl")
        paramPack.save_to_file(p_file)
    
        # write sbatch
        jobName = f"dataFit_{count}_of_{len(learningRates)}"
        sb_path = create_sbatch(jobName,template_file,p_file,inputPath,iterDir)
        
        # submit
        subprocess.run(["sbatch", sb_path])
        count = count+1
        

if __name__ == "__main__":
    
    main()