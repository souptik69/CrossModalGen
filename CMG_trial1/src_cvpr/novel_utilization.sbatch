#!/bin/bash

#SBATCH --job-name=codebook_util
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=4:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization/job_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization/job_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directory
mkdir -p $SRCDIR/CMG_trial1/Analysis/Codebook_Utilization

echo "======================================"
echo "Codebook Utilization Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "======================================"

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run analysis
python $SRCDIR/CMG_trial1/src_cvpr/novel_utilization.py

echo "======================================"
echo "Analysis Complete!"
echo "End Time: $(date)"
echo "======================================"