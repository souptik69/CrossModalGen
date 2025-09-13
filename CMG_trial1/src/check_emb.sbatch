#!/bin/bash

#SBATCH --job-name=Codebook_Viz
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --time=1:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Check_MOSEI_class/codebook_viz_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Check_MOSEI_class/codebook_viz_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm/0_Check_MOSEI_class
mkdir -p $SRCDIR/Misc/visualizations/256codebook_Norm_seq10_10_unsupervised_reg_TAV_L2_test

source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Configuration
CHECKPOINT_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/MOSEI_Models2/256codebook_Norm_seq10_10_unsupervised_reg_TAV_L2_test/checkpoint_1/MOSEI-model-unsupervised-14.pt"
OUTPUT_DIR="/project/ag-jafra/Souptik/CMG_New/Experiments/Misc/visualizations/256codebook_Norm_seq10_10_unsupervised_reg_TAV_L2_test"
TOP_K=50

echo "Starting codebook visualization..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Top K vectors: $TOP_K"

# Run visualization
python $SRCDIR/CMG_trial1/src/check_emb.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --top_k $TOP_K

echo "Codebook visualization complete!"