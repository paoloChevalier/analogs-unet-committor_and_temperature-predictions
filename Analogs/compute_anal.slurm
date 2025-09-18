#!/bin/bash
#SBATCH --partition=zen16
#SBATCH --time=4:00:00
#SBATCH --mem=240G

module purge
module load pangeo-meso/2025.01.24

python compute_analogs.py
