#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVT_Novel_CapacityFairness/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVT_Novel_CapacityFairness/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/AVT_Novel_CapacityFairness
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Novel_Models/AVT_CapacityFairness/90k/checkpoint
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Novel_Models/AVT_CapacityFairness/40k/checkpoint


# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new


# Run pretraining
# python $SRCDIR/CMG_trial1/src_cvpr/pretrain_novel_AVT.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Novel_Models/AVT_CapacityFairness/90k/" \
#     --n_epoch 6 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "vggsound_AVT_90k" \
#     --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Novel_Models/AVT_CapacityFairness/90k/checkpoint/" \
#     --print_freq 1

python $SRCDIR/CMG_trial1/src_cvpr/pretrain_novel_AVT.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Novel_Models/AVT_CapacityFairness/40k/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_40k" \
    --model_save_path "$SRCDIR/CMG_trial1/Benchmarks/Novel_Models/AVT_CapacityFairness/40k/checkpoint/" \
    --print_freq 1
    
echo "Pretraining complete!"