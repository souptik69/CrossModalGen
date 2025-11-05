#!/bin/bash

#SBATCH --job-name=AV_ablation
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AV_ablation/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AV_ablation/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/AV_ablation


# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/AlignEMA_HIER/checkpoint

python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AV.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/AlignEMA_HIER/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/AlignEMA_HIER/checkpoint/" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "AlignEMA" \
    --print_freq 1


# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/HIER/checkpoint

# python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AV.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/HIER/" \
#     --n_epoch 6 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "vggsound_AVT_40k" \
#     --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/HIER/checkpoint/" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "HIER" \
#     --print_freq 1


# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/Reset/checkpoint

# python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AV.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/Reset/" \
#     --n_epoch 6 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "vggsound_AVT_40k" \
#     --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/Reset/checkpoint/" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "Reset" \
#     --print_freq 1


# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/CMCM/checkpoint

# python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AV.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/CMCM/" \
#     --n_epoch 6 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "vggsound_AVT_40k" \
#     --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/CMCM/checkpoint/" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CMCM" \
#     --print_freq 1


# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/CPC/checkpoint

# python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AV.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/CPC/" \
#     --n_epoch 6 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "vggsound_AVT_40k" \
#     --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AV/CPC/checkpoint/" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CPC" \
#     --print_freq 1


echo "Pretraining complete!"