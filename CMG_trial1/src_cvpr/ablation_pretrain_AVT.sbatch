#!/bin/bash

#SBATCH --job-name=AVT_ablation
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=40:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVT_ablation_New/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVT_ablation_New/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/AVT_ablation_New


# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new


mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/VAT/checkpoint

python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/VAT/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/VAT/checkpoint/" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "VAT" \
    --print_freq 1

mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/VTA/checkpoint

python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/VTA/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/VTA/checkpoint/" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "VTA" \
    --print_freq 1

mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/ATV/checkpoint

python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/ATV/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/ATV/checkpoint/" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "ATV" \
    --print_freq 1

mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/AVT/checkpoint

python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/AVT/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/AVT/checkpoint/" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "AVT" \
    --print_freq 1


# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/HIER/checkpoint

# python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/HIER/" \
#     --n_epoch 6 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "vggsound_AVT_40k" \
#     --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/HIER/checkpoint/" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "HIER" \
#     --print_freq 1


# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/Reset/checkpoint

# python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/Reset/" \
#     --n_epoch 6 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "vggsound_AVT_40k" \
#     --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/Reset/checkpoint/" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "Reset" \
#     --print_freq 1


# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/CMCM/checkpoint

# python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/CMCM/" \
#     --n_epoch 6 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "vggsound_AVT_40k" \
#     --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/CMCM/checkpoint/" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CMCM" \
#     --print_freq 1


# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/CPC/checkpoint

# python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/CPC/" \
#     --n_epoch 6 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "vggsound_AVT_40k" \
#     --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/CPC/checkpoint/" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CPC" \
#     --print_freq 1


echo "Pretraining complete!"