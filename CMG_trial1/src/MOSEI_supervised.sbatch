#!/bin/bash

#SBATCH --job-name=MOSEI_supervised_class
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=3:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/128dim_256_Dynamic_Padding_reg_50_MOSEI_supervised_TAV_L2/mosei_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/128dim_256_Dynamic_Padding_reg_50_MOSEI_supervised_TAV_L2/mosei_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/128dim_256_Dynamic_Padding_reg_50_MOSEI_supervised_TAV_L2
mkdir -p $SRCDIR/CMG_trial1/MOSEI_Tests_1/128dim_256_Dynamic_Padding_reg_50_MOSEI_supervised_TAV_L2_14

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Define experiment configurations
# Since this is testing only, we don't need training parameters

# Experiment 1: MSR Mode (Multimodal Sentiment Regression)
echo "Running MSR Mode - Multimodal Sentiment Regression Testing"
python $SRCDIR/CMG_trial1/src/MOSEI_supervised_test.py \
    --gpu 0 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests_1/128dim_256_Dynamic_Padding_reg_50_MOSEI_supervised_TAV_L2_14/MSR/" \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosei" \
    --print_freq 1 \
    --test_mode "MSR" \
    --evaluate

echo "MSR Mode testing completed!"

# mkdir -p $SRCDIR/CMG_trial1/MOSEI_Tests_1/NoLSTM_seq50_100_supervised_reg_TAV_5
# Experiment 2: CMG Mode (Cross-Modal Generalization)
echo "Running CMG Mode - Cross-Modal Generalization Testing"
python $SRCDIR/CMG_trial1/src/MOSEI_supervised_test.py \
    --gpu 0 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests_1/128dim_256_Dynamic_Padding_reg_50_MOSEI_supervised_TAV_L2_14/CMG/" \
    --batch_size 32 \
    --test_batch_size 32 \
    --dataset_name "mosei" \
    --print_freq 1 \
    --test_mode "CMG" \
    --evaluate

echo "CMG Mode testing completed!"

echo "All MOSEI supervised tests completed successfully!"

# Optional: If you want to test specific modalities in CMG mode separately
# Uncomment the sections below:

# # CMG Mode - Audio only
# echo "Running CMG Mode - Audio Testing"
# python $SRCDIR/CMG_trial1/src/MOSEI_supervised_test.py \
#     --gpu 0 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSEI_supervised/CMG_audio/" \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "mosei" \
#     --print_freq 1 \
#     --test_mode "CMG" \
#     --modality "audio" \
#     --evaluate

# # CMG Mode - Video only  
# echo "Running CMG Mode - Video Testing"
# python $SRCDIR/CMG_trial1/src/MOSEI_supervised_test.py \
#     --gpu 0 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSEI_supervised/CMG_video/" \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "mosei" \
#     --print_freq 1 \
#     --test_mode "CMG" \
#     --modality "video" \
#     --evaluate

# # CMG Mode - Text only
# echo "Running CMG Mode - Text Testing"
# python $SRCDIR/CMG_trial1/src/MOSEI_supervised_test.py \
#     --gpu 0 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Tests/MOSEI_supervised/CMG_text/" \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "mosei" \
#     --print_freq 1 \
#     --test_mode "CMG" \
#     --modality "text" \
#     --evaluate