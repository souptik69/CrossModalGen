#!/bin/bash

#SBATCH --job-name=MOSI_MOSEI_unsupervised_class
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=3:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/MOSI_MOSEI_unsupervised_class_norm/mosi_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/MOSI_MOSEI_unsupervised_class_norm/mosi_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/MOSI_MOSEI_unsupervised_class_norm
mkdir -p $SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test
mkdir -p $SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSEI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new


# Experiment 1: MSR Mode (Multimodal Sentiment Regression)
echo "Running MSR Mode - Multimodal Sentiment Regression"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/MSR/" \
    --n_epoch 60 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosi" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/MSR/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "MSR"

echo "MOSI MSR Mode completed!"

# Experiment 2: CMG Mode - Audio Training
echo "Running CMG Mode - Audio Training"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_audio/" \
    --n_epoch 60 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosi" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_audio/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "CMG" \
    --modality "audio"

echo "MOSI CMG Audio Mode completed!"

# Experiment 3: CMG Mode - Video Training
echo "Running CMG Mode - Video Training"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_video/" \
    --n_epoch 60 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosi" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_video/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "CMG" \
    --modality "video"

echo "MOSI CMG Video Mode completed!"

# Experiment 4: CMG Mode - Text Training
echo "Running CMG Mode - Text Training"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_text/" \
    --n_epoch 60 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosi" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_text/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "CMG" \
    --modality "text"

echo "MOSI CMG Text Mode completed!"

echo "All MOSI unsupervised tests completed successfully!"


# Experiment 1: MSR Mode (Multimodal Sentiment Regression)
echo "Running MSR Mode - Multimodal Sentiment Regression"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSEI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/MSR_1/" \
    --n_epoch 60 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosei" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSEI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/MSR_1/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "MSR"

echo "MOSEI MSR Mode completed!"

# Experiment 2: CMG Mode - Audio Training
echo "Running CMG Mode - Audio Training"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSEI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_audio/" \
    --n_epoch 60 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosei" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSEI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_audio/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "CMG" \
    --modality "audio"

echo "MOSEI CMG Audio Mode completed!"

# Experiment 3: CMG Mode - Video Training
echo "Running CMG Mode - Video Training"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSEI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_video/" \
    --n_epoch 60 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosei" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSEI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_video/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "CMG" \
    --modality "video"

echo "MOSEI CMG Video Mode completed!"

# Experiment 4: CMG Mode - Text Training
echo "Running CMG Mode - Text Training"
python $SRCDIR/CMG_trial1/src/MOSI_unsupervised_test.py \
    --gpu 0 \
    --lr 0.0001 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSEI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_text/" \
    --n_epoch 60 \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosei" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Tests_1/MOSEI_256codebook_Norm_seq20_20_unsupervised_reg_TAV_L2_test/CMG_text/" \
    --print_freq 1 \
    --eval_freq 1 \
    --test_mode "CMG" \
    --modality "text"

echo "MOSEI CMG Text Mode completed!"

echo "All MOSEI unsupervised tests completed successfully!"