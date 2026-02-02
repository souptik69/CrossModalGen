#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Ablation_Results_AVT_New/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Ablation_Results_AVT_New/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new




mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Ablation_Results_AVT_New
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VAT/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VAT/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VAT/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VAT/AVVP/V2A

CHECKPOINT_PATH_AVT_VAT="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/VAT/checkpoint/Ablation-AVT-5.pt"

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VAT/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "VAT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_VAT" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VAT/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "VAT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_VAT" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VAT/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "VAT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_VAT" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VAT/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "VAT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_VAT" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"




mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VTA/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VTA/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VTA/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VTA/AVVP/V2A

CHECKPOINT_PATH_AVT_VTA="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/VTA/checkpoint/Ablation-AVT-5.pt"

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VTA/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "VTA" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_VTA" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VTA/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "VTA" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_VTA" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VTA/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "VTA" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_VTA" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/VTA/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "VTA" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_VTA" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"


mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/ATV/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/ATV/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/ATV/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/ATV/AVVP/V2A

CHECKPOINT_PATH_AVT_ATV="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/ATV/checkpoint/Ablation-AVT-5.pt"

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/ATV/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "ATV" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_ATV" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/ATV/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "ATV" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_ATV" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/ATV/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "ATV" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_ATV" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/ATV/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "ATV" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_ATV" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"



mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/AVT/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/AVT/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/AVT/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/AVT/AVVP/V2A

CHECKPOINT_PATH_AVT_AVT="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/AVT/checkpoint/Ablation-AVT-5.pt"

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/AVT/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_AVT" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/AVT/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_AVT" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/AVT/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_AVT" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/AVT/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --ablation "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_AVT" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"



# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/CPC/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/CPC/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/CPC/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/CPC/AVVP/V2A

# CHECKPOINT_PATH_AVT_CPC="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain_1/AVT/CPC/checkpoint/Ablation-AVT-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/CPC/AVE/A2V/" \
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
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/CPC/AVE/V2A/" \
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
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/CPC/AVVP/V2A/" \
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
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results_1/AVT/CPC/AVVP/A2V/" \
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



