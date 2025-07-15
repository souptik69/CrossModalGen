#!/bin/bash

#SBATCH --job-name=Detailed_Inspect_MOSEI_MOSI
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=2:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_MultiBench_1/detailed_inspect_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_MultiBench_1/detailed_inspect_%j.err

# Project and source directories
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directory for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/0_MultiBench_1

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Print job information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo "========================================="

# Run detailed data inspection
echo "Starting detailed data inspection..."
python $SRCDIR/CMG_trial1/src/mosei_dataset/mosei_test.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Data inspection completed successfully!"
    echo "End Time: $(date)"
    echo "========================================="
else
    echo "========================================="
    echo "Error: Data inspection failed!"
    echo "End Time: $(date)"
    echo "========================================="
    exit 1
fi