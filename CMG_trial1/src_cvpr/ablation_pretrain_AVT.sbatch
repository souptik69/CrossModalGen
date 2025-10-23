#!/bin/bash

#SBATCH --job-name=AVT_ablation
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVT_ablation/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVT_ablation/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/AVT_ablation


# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/128Codebook/checkpoint

python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/128Codebook/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/128Codebook/checkpoint/" \
    --n_embeddings 128 \
    --embedding_dim 256 \
    --print_freq 1


mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/256Codebook/checkpoint

python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/256Codebook/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/256Codebook/checkpoint/" \
    --n_embeddings 256 \
    --embedding_dim 256 \
    --print_freq 1


mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/400Codebook/checkpoint

python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/400Codebook/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/400Codebook/checkpoint/" \
    --n_embeddings 400 \
    --embedding_dim 256 \
    --print_freq 1


mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/512Codebook/checkpoint

python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/512Codebook/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/512Codebook/checkpoint/" \
    --n_embeddings 512 \
    --embedding_dim 256 \
    --print_freq 1


mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/800Codebook/checkpoint

python $SRCDIR/CMG_trial1/src_cvpr/ablation_pretrain_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/800Codebook/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/800Codebook/checkpoint/" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --print_freq 1


echo "Pretraining complete!"