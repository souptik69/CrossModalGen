#!/bin/bash

#SBATCH --job-name=AVVP_train
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu003
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=165:00:00
#SBATCH --output=/project/ag-jafra/Souptik/VGGSoundAVEL/Slurm/AVVP_av_accuracy40k/avvp_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/VGGSoundAVEL/Slurm/AVVP_av_accuracy40k/avvp_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/VGGSoundAVEL

# Create output directories for logs
mkdir -p $SRCDIR/Slurm/AVVP_av_accuracy40k

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/VGGSoundAVEL/CMG/envs/CMG_final

# Run AVVP training
python $SRCDIR/CMG/code/src/avvp.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_Models/AVVP_Exp/accuracy_40k/original/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --print_freq 1 \
    --eval_freq 1

echo "AVVP training complete!"