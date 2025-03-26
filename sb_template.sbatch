#!/bin/bash
#SBATCH --job-name={jobName}
#SBATCH --output={outputDir}/out.log
#SBATCH --error={outputDir}/err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=20:00:00
#SBATCH --mem=8GB

module purge

srun singularity exec --nv --overlay /home/km3911/code/pytorch/my_pytorch.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh;
	python data_fit_20250325.py --config_path {paramFile} --input_path {dataDir} --output_path {outputDir}“
