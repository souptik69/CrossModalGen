#!/bin/bash

#SBATCH --job-name=AVVP_parallel_train
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=65:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0110_AVVP_Novel_Parallel/avvp_parallel_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0110_AVVP_Novel_Parallel/avvp_parallel_%j.err

# Set source directory (fill in your source directory path)
SRCDIR="/project/ag-jafra/Souptik/CMG_New/Experiments"

# Set paths for each experiment (fill in your desired paths)
AVVP_VA_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/AVVP_Tests/FixMeta/90k_acc/ND_V2A/"
AVVP_AV_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/AVVP_Tests/FixMeta/90k_acc/ND_A2V/"
AVVP_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/AVVP_Tests/FixMeta/90k_acc/ND_Event/"

# Set GPU IDs (specify which GPUs to use, e.g., 0,1,2)
GPU_0="leinevmgpu005"
GPU_1="leinevmgpu006"
GPU_2="leinevmgpu008"

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/0110_AVVP_Novel_Parallel

# Create experiment directories
mkdir -p "$AVVP_VA_PATH"
mkdir -p "$AVVP_AV_PATH" 
mkdir -p "$AVVP_PATH"

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

echo "Starting parallel AVVP training on 3 GPUs..."
echo "AVVP_VA on GPU $GPU_0"
echo "AVVP_AV on GPU $GPU_1" 
echo "AVVP on GPU $GPU_2"

# Run AVVP_VA training on specified GPU
CUDA_VISIBLE_DEVICES=$GPU_0 python $SRCDIR/CMG_trial1/src/avvp_novel.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$AVVP_VA_PATH" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --model_save_path "$AVVP_VA_PATH" \
    --print_freq 1 \
    --eval_freq 1 > "$AVVP_VA_PATH/training.log" 2>&1 &
PID1=$!

# Run AVVP_AV training on specified GPU
CUDA_VISIBLE_DEVICES=$GPU_1 python $SRCDIR/CMG_trial1/src/avvp_novel.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$AVVP_AV_PATH" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --model_save_path "$AVVP_AV_PATH" \
    --print_freq 1 \
    --eval_freq 1 > "$AVVP_AV_PATH/training.log" 2>&1 &
PID2=$!
# Run AVVP training on specified GPU
CUDA_VISIBLE_DEVICES=$GPU_2 python $SRCDIR/CMG_trial1/src/avvp_novel.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$AVVP_PATH" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp" \
    --model_save_path "$AVVP_PATH" \
    --print_freq 1 \
    --eval_freq 1 > "$AVVP_PATH/training.log" 2>&1 &

PID3=$!

echo "Started AVVP_VA training with PID: $PID1"
echo "Started AVVP_AV training with PID: $PID2"
echo "Started AVVP training with PID: $PID3"

# Wait for all background jobs to complete
wait $PID1
echo "AVVP_VA training completed"

wait $PID2  
echo "AVVP_AV training completed"

wait $PID3
echo "AVVP training completed"

echo "All AVVP training jobs completed successfully!"