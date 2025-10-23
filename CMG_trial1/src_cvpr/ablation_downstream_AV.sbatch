#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Ablation_Results_AV/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Ablation_Results_AV/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new




mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Codebook/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Codebook/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Codebook/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Codebook/AVVP/V2A

CHECKPOINT_PATH_AV_128Codebook=

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Codebook/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AV" \
    --n_embeddings 128 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_128Codebook" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Codebook/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AV" \
    --n_embeddings 128 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_128Codebook" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Codebook/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AV" \
    --n_embeddings 128 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_128Codebook" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Codebook/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AV" \
    --n_embeddings 128 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_128Codebook" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"





mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/256Codebook/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/256Codebook/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/256Codebook/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/256Codebook/AVVP/V2A

CHECKPOINT_PATH_AV_256Codebook=

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/256Codebook/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AV" \
    --n_embeddings 256 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_256Codebook" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/256Codebook/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AV" \
    --n_embeddings 256 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_256Codebook" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/256Codebook/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AV" \
    --n_embeddings 256 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_256Codebook" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/256Codebook/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AV" \
    --n_embeddings 256 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_256Codebook" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"


mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/400Codebook/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/400Codebook/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/400Codebook/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/400Codebook/AVVP/V2A

CHECKPOINT_PATH_AV_400Codebook=

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/400Codebook/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AV" \
    --n_embeddings 400 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_400Codebook" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/400Codebook/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AV" \
    --n_embeddings 400 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_400Codebook" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/400Codebook/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AV" \
    --n_embeddings 400 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_400Codebook" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/400Codebook/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AV" \
    --n_embeddings 400 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_400Codebook" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"



mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Codebook/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Codebook/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Codebook/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Codebook/AVVP/V2A

CHECKPOINT_PATH_AV_512Codebook=

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Codebook/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AV" \
    --n_embeddings 512 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_512Codebook" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Codebook/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AV" \
    --n_embeddings 512 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_512Codebook" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Codebook/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AV" \
    --n_embeddings 512 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_512Codebook" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Codebook/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AV" \
    --n_embeddings 512 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_512Codebook" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"




mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Codebook/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Codebook/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Codebook/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Codebook/AVVP/V2A

CHECKPOINT_PATH_AV_800Codebook=

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Codebook/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_800Codebook" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Codebook/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_800Codebook" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Codebook/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_800Codebook" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Codebook/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_800Codebook" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"



