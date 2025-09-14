#!/bin/bash

#SBATCH --job-name=Test_MOSEI_MOSI_DataLoaders
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=02:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/00_test_dataloaders_2/test_output_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/00_test_dataloaders_2/test_error_%j.err

# Print job information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo "========================================="

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directory
mkdir -p $SRCDIR/CMG_trial1/slurm/00_test_dataloaders_2

# Source bashrc and activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

echo "Starting comprehensive dataloader testing..."

# Run the dataloader tests
python $SRCDIR/CMG_trial1/src/dataset/MOSEI_MOSI.py

echo "========================================="
echo "Dataloader testing completed!"
echo "End Time: $(date)"
echo "========================================="