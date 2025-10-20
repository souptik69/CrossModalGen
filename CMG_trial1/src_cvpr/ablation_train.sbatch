#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVT_ablation_256Codebook/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVT_ablation_256Codebook/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/AVT_ablation_256Codebook
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation/AVT/256Codebook/checkpoint
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation/AV/256Codebook/checkpoint


# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new


python $SRCDIR/CMG_trial1/src_cvpr/pretrain_novel_AV.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation/AV/256Codebook/" \
    --n_epoch 5 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation/AV/256Codebook/checkpoint/" \
    --print_freq 1

python $SRCDIR/CMG_trial1/src_cvpr/pretrain_novel_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation/AVT/256Codebook/" \
    --n_epoch 5 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Ablation/AVT/256Codebook/checkpoint/" \
    --print_freq 1

echo "Pretraining complete!"