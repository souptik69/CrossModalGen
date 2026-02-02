#!/bin/bash

#SBATCH --job-name=Novel_retrieval_benchmark
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=02:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Retrieval_Novel/Retrieval_Novel_1_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Retrieval_Novel/Retrieval_Novel_1_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories
mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Retrieval_Novel
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_Novel_1/AVT/40k/Clotho
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_Novel_1/AVT/40k/MSCOCO
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_Novel_1/AVT/90k/Clotho
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_Novel_1/AVT/90k/MSCOCO

# Checkpoint paths - Novel Model
CHECKPOINT_PATH_AVT_NOVEL_40k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Novel_Models/AVT/40k/checkpoint/Ablation-AVT-5.pt"
CHECKPOINT_PATH_AVT_NOVEL_90k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Novel_Models/AVT/90k/checkpoint/HierVQ-model-AVT-5.pt"

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# ===================== Novel AVT 40k =====================

# # Clotho (Audio-Text retrieval)
# echo "Running Novel AVT 40k - Clotho..."
# python $SRCDIR/CMG_trial1/src_cvpr/Retrieval_Novel_1.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_Novel_1/AVT/40k/Clotho/" \
#     --n_epoch 7 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "clotho" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_NOVEL_40k" \
#     --print_freq 1 \
#     --eval_freq 1

# echo "Novel AVT 40k Clotho complete!"

# MSCOCO (Video-Text retrieval)
# echo "Running Novel AVT 40k - MSCOCO..."
# python $SRCDIR/CMG_trial1/src_cvpr/retrieval_novel.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_Novel_1/AVT/40k/MSCOCO/" \
#     --n_epoch 7 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "mscoco" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_NOVEL_40k" \
#     --print_freq 1 \
#     --eval_freq 1

# echo "Novel AVT 40k MSCOCO complete!"

# ===================== Novel AVT 90k =====================

# # Clotho (Audio-Text retrieval)
# echo "Running Novel AVT 90k - Clotho..."
# python $SRCDIR/CMG_trial1/src_cvpr/Retrieval_Novel_1.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_Novel_1/AVT/90k/Clotho/" \
#     --n_epoch 7 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "clotho" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_NOVEL_90k" \
#     --print_freq 1 \
#     --eval_freq 1

# echo "Novel AVT 90k Clotho complete!"

# MSCOCO (Video-Text retrieval)
echo "Running Novel AVT 90k - MSCOCO..."
python $SRCDIR/CMG_trial1/src_cvpr/retrieval_novel.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_Novel_1/AVT/90k/MSCOCO/" \
    --n_epoch 7 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "mscoco" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_NOVEL_90k" \
    --print_freq 1 \
    --eval_freq 1

echo "Novel AVT 90k MSCOCO complete!"

echo "All Novel model retrieval benchmarks complete!"
