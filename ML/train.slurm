#!/bin/bash

# SBATCH instructions
#SBATCH --chdir=/net/nfs/ssd2/pchevali/train_models
#SBATCH --partition=batch
#SBATCH --job-name=unet_training
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mail-user=paolo.chevalier@lsce.ipsl.fr
#SBATCH --mail-type=ALL
#SBATCH --error=job-%j.log
#SBATCH --output=job-%j.log

# Load system profile and purge previous modules
source /etc/profile
module purge

# Load TensorFlow module
module load 'tensorflow/2.15.0'

# Enable unbuffered Python output
export PYTHONUNBUFFERED=1

# Run all Python scripts sequentially
python example.py


