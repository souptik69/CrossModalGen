#!/bin/bash

#SBATCH --job-name=Pretrain_Viz
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu008
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=15:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Check_training_hier_2/pretrain_viz_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Check_training_hier_2/pretrain_viz_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create necessary directories
mkdir -p $SRCDIR/CMG_trial1/slurm/0_Check_training_hier_2
mkdir -p $SRCDIR/Misc/visualizations_hier_1/1_training_evolution

source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Configuration
CHECKPOINT_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/AVT_model/Best_Text_CPC_noNoise/40k/checkpoint/DCID-model-0.pt"
OUTPUT_DIR="/project/ag-jafra/Souptik/CMG_New/Experiments/Misc/visualizations_hier_1/1_training_evolution"
TOP_K=25
BATCH_SIZE=64
LEARNING_RATE=0.0004
GPU_ID=0

echo "Starting pretraining with codebook visualization..."
echo "========================================================"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Top K vectors to track: $TOP_K"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "GPU ID: $GPU_ID"
echo "========================================================"

# Run pretraining with visualization
python $SRCDIR/CMG_trial1/src/pretrain_viz.py \
    --gpu $GPU_ID \
    --lr $LEARNING_RATE \
    --clip_gradient 0.5 \
    --batch_size $BATCH_SIZE \
    --dataset_name "vggsound" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --top_k $TOP_K \
    --print_freq 1

echo "========================================================"
echo "Pretraining with visualization complete!"
echo "Visualizations saved to: $OUTPUT_DIR"
echo "Check the following files:"
echo "  - combined_modalities_iter_0.png"
echo "  - combined_modalities_iter_20.png"
echo "  - combined_modalities_iter_40.png"
echo "  - combined_modalities_iter_80.png"
echo "  - combined_modalities_iter_120.png"
echo "  - combined_modalities_iter_320.png"
echo "  - combined_modalities_iter_620.png"
echo "========================================================"