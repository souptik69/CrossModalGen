#!/bin/bash

#SBATCH --job-name=Test_Adaptive_CPC
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=01:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/01_test_adaptive_cpc/test_output_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/01_test_adaptive_cpc/test_error_%j.err

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

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/model

# Create output directory
mkdir -p $SRCDIR/CMG_trial1/slurm/01_test_adaptive_cpc

# Source bashrc and activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

echo "Starting adaptive CPC testing..."

# Run the test script
python $SRCDIR/Test_CPC.py

echo "========================================="
echo "Adaptive CPC testing completed!"
echo "End Time: $(date)"
echo "========================================="