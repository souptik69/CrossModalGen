#!/bin/bash

#SBATCH --job-name=Codebook_Viz
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu008
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Check_MOSEI/codebook_viz_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Check_MOSEI/codebook_viz_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm/0_Check_MOSEI
mkdir -p $SRCDIR/Misc/visualizations/MOSEI_VA_9

source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Configuration
CHECKPOINT_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/MOSEI_Models/mosei_supervised/checkpoint/MOSEI-model-9.pt"
OUTPUT_DIR="/project/ag-jafra/Souptik/CMG_New/Experiments/Misc/visualizations/MOSEI_VA_9"
TOP_K=25

echo "Starting codebook visualization..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Top K vectors: $TOP_K"s

# Run visualization
python $SRCDIR/CMG_trial1/src/check_emb.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --top_k $TOP_K

echo "Codebook visualization complete!"