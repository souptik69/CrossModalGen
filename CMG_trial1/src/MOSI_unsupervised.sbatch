#!/bin/bash

#SBATCH --job-name=MOSI_unsupervised_test
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu007
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=65:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/MOSI_unsupervised_test/mosi_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/MOSI_unsupervised_test/mosi_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/MOSI_unsupervised_test

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Define experiment configurations
# You can modify these variables to run different experiments

# Experiment 1: MSR Mode (Multimodal Sentiment Regression)
echo "Running MSR Mode - Multimodal Sentiment Regression"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSI_unsupervised/MSR_AV/" \
    --n_epoch 30 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosi" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSI_unsupervised/MSR_AV/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "MSR"

echo "MSR Mode completed!"

# Experiment 2: CMG Mode - Audio Training
echo "Running CMG Mode - Audio Training"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSI_unsupervised/CMG_audio_AV/" \
    --n_epoch 30 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosi" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSI_unsupervised/CMG_audio_AV/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "CMG" \
    --modality "audio"

echo "CMG Audio Mode completed!"

# Experiment 3: CMG Mode - Video Training
echo "Running CMG Mode - Video Training"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSI_unsupervised/CMG_video_AV/" \
    --n_epoch 30 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosi" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSI_unsupervised/CMG_video_AV/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "CMG" \
    --modality "video"

echo "CMG Video Mode completed!"

# Experiment 4: CMG Mode - Text Training
echo "Running CMG Mode - Text Training"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSI_unsupervised/CMG_text_AV/" \
    --n_epoch 30 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosi" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSI_unsupervised/CMG_text_AV/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "CMG" \
    --modality "text"

echo "CMG Text Mode completed!"

echo "All MOSI unsupervised tests completed successfully!"