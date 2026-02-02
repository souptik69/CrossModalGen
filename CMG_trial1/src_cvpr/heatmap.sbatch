#!/bin/bash

#SBATCH --job-name=codebook_heatmap
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=3:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization/heatmap_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization/heatmap_%j.err

# Create output directory
mkdir -p /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization

echo "======================================"
echo "CoDAAR AV Intensity Heatmap Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "======================================"

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run analysis
python /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/codebook_intensity_av.py

echo ""
echo "======================================"
echo "Complete!"
echo "End Time: $(date)"
echo "======================================"