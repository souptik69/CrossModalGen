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
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Hier_VA/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Hier_VA/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Hier_VA/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Hier_VA/AVVP/V2A

CHECKPOINT_PATH_AV_Hier_VA="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/Hier_VA/checkpoint/HierVQ-model-AV-5.pt"

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Hier_VA/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_Hier_VA" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Hier_VA/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_Hier_VA" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Hier_VA/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_Hier_VA" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Hier_VA/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AV_Hier_VA" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"





# mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/64Dimensions/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/64Dimensions/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/64Dimensions/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/64Dimensions/AVVP/V2A

# CHECKPOINT_PATH_AV_64Dimensions="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/64Dimensions/checkpoint/HierVQ-model-AV-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/64Dimensions/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 64 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_64Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/64Dimensions/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 64 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_64Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/64Dimensions/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 64 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_64Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/64Dimensions/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 64 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_64Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"


# mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Dimensions/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Dimensions/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Dimensions/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Dimensions/AVVP/V2A

# CHECKPOINT_PATH_AV_128Dimensions="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/128Dimensions/checkpoint/HierVQ-model-AV-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Dimensions/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 128 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_128Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Dimensions/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 128 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_128Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Dimensions/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 128 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_128Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/128Dimensions/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 128 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_128Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"



# mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Dimensions/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Dimensions/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Dimensions/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Dimensions/AVVP/V2A

# CHECKPOINT_PATH_AV_512Dimensions="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/512Dimensions/checkpoint/HierVQ-model-AV-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Dimensions/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AV" \
#     --n_embeddings 512 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_512Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Dimensions/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AV" \
#     --n_embeddings 512 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_512Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Dimensions/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AV" \
#     --n_embeddings 512 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_512Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/512Dimensions/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AV" \
#     --n_embeddings 512 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_512Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"




# mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Dimensions/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Dimensions/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Dimensions/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Dimensions/AVVP/V2A

# CHECKPOINT_PATH_AV_800Dimensions="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/800Dimensions/checkpoint/HierVQ-model-AV-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Dimensions/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_800Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Dimensions/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_800Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Dimensions/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_800Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/800Dimensions/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_800Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"



