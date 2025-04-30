#!/bin/bash

#SBATCH --job-name=AVT_100k_test
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu008
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=48:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/AVT_100k_Eval/test_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/AVT_100k_Eval/test_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/AVT_100k_Eval

# Set result output directory
OUTPUT_DIR=$SRCDIR/CMG_trial1/Results/AVT_100k_Test
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
echo "Testing started at: $(date)"

# Run the test script
python $SRCDIR/CMG_trial1/src/test_avt_vqvae.py \
    --gpu 0 \
    --batch_size 64 \
    --output_dir "$OUTPUT_DIR" \
    --snapshot_pref "$OUTPUT_DIR/logs" \
    --dataset_name "vggsound_AVT" \
    --print_freq 10

echo "Testing complete at: $(date)"
echo "Results saved to $OUTPUT_DIR"