#!/bin/bash

#SBATCH --job-name=AVT_retrieval_benchmark
#SBATCH --partition=leinegpu_plri
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=04:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Retrieval_Benchmark/retrieval_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Retrieval_Benchmark/retrieval_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories
mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Retrieval_Benchmark
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_MICU_1/AVT/40k/Clotho
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_MICU_1/AVT/40k/MSCOCO
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_MICU_1/AVT/90k/Clotho
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_MICU_1/AVT/90k/MSCOCO

# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_DCID_1/AVT/40k/Clotho
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_DCID_1/AVT/40k/MSCOCO
# mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_DCID_1/AVT/90k/Clotho
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Retrieval_DCID_1/AVT/90k/MSCOCO

# Checkpoint paths
CHECKPOINT_PATH_AVT_MICU_40k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/MICU/40k/MICU.pt"
CHECKPOINT_PATH_AVT_MICU_90k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/MICU/90k/checkpoint/MICU-step5400.pt"


# Checkpoint paths - DCID
CHECKPOINT_PATH_AVT_DCID_40k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/DCID/40k/AVT_vgg40k_size400.pt"
CHECKPOINT_PATH_AVT_DCID_90k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/DCID/90k/checkpoint/DCID-model-4.pt"

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/VGGSoundAVEL/CMG/envs/CMG_final

# ===================== AVT 40k =====================

# Clotho (Audio-Text retrieval)
# echo "Running MICU AVT 40k - Clotho..."
# python $SRCDIR/CMG_trial1/src_cvpr/retrieval_benchmark.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_MICU_1/AVT/40k/Clotho/" \
#     --n_epoch 7 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "clotho" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_MICU_40k" \
#     --print_freq 1 \
#     --eval_freq 1

# echo "MICU AVT 40k Clotho complete!"

# # MSCOCO (Video-Text retrieval)
# echo "Running MICU AVT 40k - MSCOCO..."
# python $SRCDIR/CMG_trial1/src_cvpr/retrieval_benchmark.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_MICU_1/AVT/40k/MSCOCO/" \
#     --n_epoch 7 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "mscoco" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_MICU_40k" \
#     --print_freq 1 \
#     --eval_freq 1

# echo "MICU AVT 40k MSCOCO complete!"

# ===================== AVT 90k =====================

# Clotho (Audio-Text retrieval)
# echo "Running MICU AVT 90k - Clotho..."
# python $SRCDIR/CMG_trial1/src_cvpr/retrieval_benchmark.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_MICU_1/AVT/90k/Clotho/" \
#     --n_epoch 7 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "clotho" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_MICU_90k" \
#     --print_freq 1 \
#     --eval_freq 1

# echo "MICU AVT 90k Clotho complete!"

# MSCOCO (Video-Text retrieval)
echo "Running MICU AVT 90k - MSCOCO..."
python $SRCDIR/CMG_trial1/src_cvpr/retrieval_benchmark.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_MICU_1/AVT/90k/MSCOCO/" \
    --n_epoch 7 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "mscoco" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_MICU_90k" \
    --print_freq 1 \
    --eval_freq 1

echo "MICU AVT 90k MSCOCO complete!"


# echo "Running DCID AVT 40k - Clotho..."
# python $SRCDIR/CMG_trial1/src_cvpr/retrieval_benchmark.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_DCID_1/AVT/40k/Clotho/" \
#     --n_epoch 7 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "clotho" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_DCID_40k" \
#     --print_freq 1 \
#     --eval_freq 1

# echo "DCID AVT 40k Clotho complete!"

# # MSCOCO (Video-Text retrieval)
# echo "Running DCID AVT 40k - MSCOCO..."
# python $SRCDIR/CMG_trial1/src_cvpr/retrieval_benchmark.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_DCID_1/AVT/40k/MSCOCO/" \
#     --n_epoch 7 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "mscoco" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_DCID_40k" \
#     --print_freq 1 \
#     --eval_freq 1

# echo "DCID AVT 40k MSCOCO complete!"

# ===================== DCID AVT 90k =====================

# Clotho (Audio-Text retrieval)
# echo "Running DCID AVT 90k - Clotho..."
# python $SRCDIR/CMG_trial1/src_cvpr/retrieval_benchmark.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_DCID_1/AVT/90k/Clotho/" \
#     --n_epoch 7 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "clotho" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_DCID_90k" \
#     --print_freq 1 \
#     --eval_freq 1

# echo "DCID AVT 90k Clotho complete!"

# MSCOCO (Video-Text retrieval)
echo "Running DCID AVT 90k - MSCOCO..."
python $SRCDIR/CMG_trial1/src_cvpr/retrieval_benchmark.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Retrieval_DCID_1/AVT/90k/MSCOCO/" \
    --n_epoch 7 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "mscoco" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_DCID_90k" \
    --print_freq 1 \
    --eval_freq 1

echo "DCID AVT 90k MSCOCO complete!"



echo "All retrieval benchmarks complete!"