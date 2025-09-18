#!/bin/bash
#SBATCH --partition=zen16
#SBATCH --time=24:00:00
#SBATCH --mem=240G

module purge
module load pangeo-meso/2025.01.24

python train_val_sets.py
