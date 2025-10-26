#!/bin/bash

#SBATCH --job-name=Codebook_Viz
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --time=1:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/0_Check_MOSEI_class_1/codebook_viz_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/0_Check_MOSEI_class_1/codebook_viz_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/0_Check_MOSEI_class_1
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Visualizations_256

source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Configuration
CHECKPOINT_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/256Codebook/checkpoint/Ablation-AVT-5.pt"
# OUTPUT_DIR="/project/ag-jafra/Souptik/CMG_New/Experiments/Misc/visualizations/Final_MOSEI_unsupervised_VAT_30dim_NewAttention_window_20"
OUTPUT_DIR="$SRCDIR/CMG_trial1/Benchmarks/Visualizations_256"
TOP_K=256

echo "Starting codebook visualization..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Top K vectors: $TOP_K"

# Run visualization
python $SRCDIR/CMG_trial1/src_cvpr/check_emb.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --top_k $TOP_K

echo "Codebook visualization complete!"