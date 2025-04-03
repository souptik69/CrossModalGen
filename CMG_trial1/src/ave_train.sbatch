#!/bin/bash

#SBATCH --job-name=AVE_train
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu008
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=165:00:00
#SBATCH --output=/project/ag-jafra/Souptik/VGGSoundAVEL/Slurm/AVE_40k_VA_precision/ave_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/VGGSoundAVEL/Slurm/AVE_40k_VA_precision/ave_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/VGGSoundAVEL

# Create output directories for logs
mkdir -p $SRCDIR/Slurm/AVE_40k_VA_precision

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/VGGSoundAVEL/CMG/envs/CMG_final

# Run AVE training
python $SRCDIR/CMG/code/src/ave_precision.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_Models/AVE_discarded/V2A_3epoch/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --print_freq 1 \
    --eval_freq 1

echo "AVE training complete!"