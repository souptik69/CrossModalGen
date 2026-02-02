#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Ablation_Results_AV_New/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Ablation_Results_AV_New/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new




mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV_New
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/EqualHIER/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/EqualHIER/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/EqualHIER/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/EqualHIER/AVVP/V2A

CHECKPOINT_PATH_AV_EqualHIER="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/EqualHIER/checkpoint/HierVQ-model-AV-5.pt"

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/EqualHIER/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "EqualHIER" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_EqualHIER" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/EqualHIER/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "EqualHIER" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_EqualHIER" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/EqualHIER/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "EqualHIER" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_EqualHIER" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/EqualHIER/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "EqualHIER" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_EqualHIER" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"





mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV_New
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/UniRecon/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/UniRecon/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/UniRecon/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/UniRecon/AVVP/V2A

CHECKPOINT_PATH_AV_UniRecon="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/UniRecon/checkpoint/HierVQ-model-AV-5.pt"

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/UniRecon/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "UniRecon" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_UniRecon" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/UniRecon/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "UniRecon" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_UniRecon" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/UniRecon/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "UniRecon" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_UniRecon" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/UniRecon/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "UniRecon" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_UniRecon" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"


mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV_New
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Order/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Order/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Order/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Order/AVVP/V2A

CHECKPOINT_PATH_AV_Order="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/Order/checkpoint/HierVQ-model-AV-5.pt"

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Order/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "Order" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_Order" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Order/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "Order" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_Order" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Order/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "Order" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_Order" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/Order/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AV" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "Order" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_Order" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"



# mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV_New
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CMCM/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CMCM/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CMCM/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CMCM/AVVP/V2A

# CHECKPOINT_PATH_AV_CMCM="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/CMCM/checkpoint/HierVQ-model-AV-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CMCM/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CMCM" \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_CMCM" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CMCM/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CMCM" \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_CMCM" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CMCM/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CMCM" \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_CMCM" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CMCM/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CMCM" \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_CMCM" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"




# mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AV_New
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CPC/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CPC/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CPC/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CPC/AVVP/V2A

# CHECKPOINT_PATH_AV_CPC="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/CPC/checkpoint/HierVQ-model-AV-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CPC/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CPC" \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_CPC" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CPC/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CPC" \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_CPC" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CPC/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CPC" \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_CPC" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AV/CPC/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AV" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CPC" \
#     --checkpoint_path "$CHECKPOINT_PATH_AV_CPC" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"



