#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
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
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Hier_TVA/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Hier_TVA/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Hier_TVA/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Hier_TVA/AVVP/V2A

CHECKPOINT_PATH_AVT_Hier_TVA="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/Hier_TVA/checkpoint/Ablation-AVT-5.pt"

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Hier_TVA/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_Hier_TVA" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Hier_TVA/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_Hier_TVA" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Hier_TVA/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_Hier_TVA" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/Hier_TVA/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AVT" \
    --n_embeddings 800 \
    --embedding_dim 256 \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_Hier_TVA" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"




# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/64Dimensions/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/64Dimensions/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/64Dimensions/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/64Dimensions/AVVP/V2A

# CHECKPOINT_PATH_AVT_64Dimensions="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/64Dimensions/checkpoint/Ablation-AVT-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/64Dimensions/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 64 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_64Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/64Dimensions/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 64 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_64Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/64Dimensions/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 64 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_64Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/64Dimensions/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 64 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_64Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"


# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/128Dimensions/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/128Dimensions/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/128Dimensions/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/128Dimensions/AVVP/V2A

# CHECKPOINT_PATH_AVT_128Dimensions="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/128Dimensions/checkpoint/Ablation-AVT-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/128Dimensions/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 128 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_128Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/128Dimensions/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 128 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_128Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/128Dimensions/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 128 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_128Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/128Dimensions/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 128 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_128Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"



# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/512Dimensions/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/512Dimensions/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/512Dimensions/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/512Dimensions/AVVP/V2A

# CHECKPOINT_PATH_AVT_512Dimensions="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/512Dimensions/checkpoint/Ablation-AVT-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/512Dimensions/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AVT" \
#     --n_embeddings 512 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_512Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/512Dimensions/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AVT" \
#     --n_embeddings 512 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_512Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/512Dimensions/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AVT" \
#     --n_embeddings 512 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_512Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/512Dimensions/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AVT" \
#     --n_embeddings 512 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_512Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"



# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/800Dimensions/AVE/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/800Dimensions/AVE/V2A
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/800Dimensions/AVVP/A2V
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/800Dimensions/AVVP/V2A

# CHECKPOINT_PATH_AVT_800Dimensions="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/800Dimensions/checkpoint/Ablation-AVT-5.pt"

# # Run AVE training
# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/800Dimensions/AVE/A2V/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_800Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/ave_ablation.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/800Dimensions/AVE/V2A/" \
#     --n_epoch 30 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_800Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/800Dimensions/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_800Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_ablation.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Ablation_Results/AVT/800Dimensions/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AVT" \
#     --n_embeddings 800 \
#     --embedding_dim 256 \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_800Dimensions" \
#     --print_freq 1 \
#     --eval_freq 1


# echo "AVVP training complete!"



