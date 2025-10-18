#!/bin/bash

#SBATCH --job-name=analyze_transformers
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=02:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm1/logs/analyze_transformers_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm1/logs/analyze_transformers_%j.err

# Create logs directory
mkdir -p /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm1/logs

source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run analysis
python /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/test_text_data.py

echo "Analysis with transformers complete!"