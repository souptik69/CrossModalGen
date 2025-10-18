#!/bin/bash

#SBATCH --job-name=test_cnt_trans
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=01:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm1/logs1/test_cnt_trans_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm1/logs1/test_cnt_trans_%j.err

# Create logs directory
mkdir -p /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm1/logs1

source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Test with transformers
python /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/test_new_cnt_transformers.py

echo "Test with transformers complete!"