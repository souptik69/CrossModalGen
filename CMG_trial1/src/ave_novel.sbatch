#!/bin/bash

#SBATCH --job-name=AVE_train
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu006
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=165:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Sum_AVE_Novel_AV_Test/ave_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Sum_AVE_Novel_AV_Test/ave_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/0_Sum_AVE_Novel_AV_Test

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run AVE training
python $SRCDIR/CMG_trial1/src/ave_new.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Novel_Model_Final/AVE_Tests/Best_sum/90k/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --model_save_path "$SRCDIR/CMG_trial1/Novel_Model_Final/AVE_Tests/Best_sum/90k/A2V/" \
    --print_freq 1 \
    --eval_freq 1

# echo "AVE training complete !!"