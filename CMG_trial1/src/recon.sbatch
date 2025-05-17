#!/bin/bash

#SBATCH --job-name=AVT_Recon_Analysis
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu006
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=24:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/AVT_Recon_Analysis/analyze_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/AVT_Recon_Analysis/analyze_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/AVT_Recon_Analysis

# Set result output directory
OUTPUT_DIR=$SRCDIR/CMG_trial1/Results/AVT_Recon_Analysis
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/visualizations
mkdir -p $OUTPUT_DIR/statistics
mkdir -p $OUTPUT_DIR/logs

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Print environment info for debugging
echo "Python: $(which python)"
echo "GPU available: $(nvidia-smi)"
echo "Analysis started at: $(date)"

# Run the analysis script
python $SRCDIR/CMG_trial1/src/recon_class_avt.py \
    --gpu 0 \
    --batch_size 64 \
    --output_dir "$OUTPUT_DIR" \
    --snapshot_pref "$OUTPUT_DIR/logs" \
    --dataset_name "vggsound_AVT" \
    --print_freq 10

echo "Analysis complete at: $(date)"
echo "Results saved to $OUTPUT_DIR"