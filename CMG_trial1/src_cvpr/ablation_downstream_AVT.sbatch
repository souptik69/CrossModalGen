#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Ablation_Results_AVT/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Ablation_Results_AVT/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new




mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AVT
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/AlignEMA_HIER/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/AlignEMA_HIER/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/AlignEMA_HIER/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/AlignEMA_HIER/AVVP/V2A

CHECKPOINT_PATH_AVT_AlignEMA="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/AlignEMA_HIER/checkpoint/Ablation-AVT-5.pt"

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/AlignEMA_HIER/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "AlignEMA" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_AlignEMA" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/AlignEMA_HIER/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "AlignEMA" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_AlignEMA" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/AlignEMA_HIER/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "AlignEMA" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_AlignEMA" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/AlignEMA_HIER/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "AlignEMA" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_AlignEMA" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"




# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/HIER/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/HIER/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/HIER/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/HIER/AVVP/V2A

# CHECKPOINT_PATH_AVT_HIER="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/HIER/checkpoint/Ablation-AVT-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/HIER/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "HIER" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_HIER" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/HIER/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "HIER" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_HIER" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/HIER/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "HIER" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_HIER" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/HIER/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "HIER" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_HIER" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"


# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Reset/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Reset/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Reset/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Reset/AVVP/V2A

# CHECKPOINT_PATH_AVT_Reset="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/Reset/checkpoint/Ablation-AVT-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Reset/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "Reset" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_Reset" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Reset/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "Reset" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_Reset" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Reset/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "Reset" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_Reset" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Reset/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "Reset" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_Reset" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"



# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CMCM/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CMCM/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CMCM/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CMCM/AVVP/V2A

# CHECKPOINT_PATH_AVT_CMCM="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/CMCM/checkpoint/Ablation-AVT-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CMCM/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CMCM" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_CMCM" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CMCM/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CMCM" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_CMCM" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CMCM/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CMCM" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_CMCM" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CMCM/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CMCM" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_CMCM" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"



# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CPC/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CPC/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CPC/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CPC/AVVP/V2A

# CHECKPOINT_PATH_AVT_CPC="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/CPC/checkpoint/Ablation-AVT-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CPC/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CPC" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_CPC" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CPC/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CPC" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_CPC" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CPC/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CPC" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_CPC" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/CPC/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --ablation "CPC" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_CPC" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"



