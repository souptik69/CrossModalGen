#!/bin/bash

#SBATCH --job-name=AVT_100k_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu005
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=165:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/AVT_100k_classloss_Test_foreground/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/AVT_100k_classloss_Test_foreground/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/AVT_100k_classloss_Test_foreground

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# export LD_LIBRARY_PATH=/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/envs/CMG_final/lib/python3.7/site-packages/torch/lib:$LD_LIBRARY_PATH
# export CUDA_HOME=/usr/local/cuda-11.7


# Run pretraining
python $SRCDIR/CMG_trial1/src/pretrain.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Models/AVT_100k_classloss_foreground/" \
    --n_epoch 6 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT" \
    --print_freq 1

echo "Pretraining complete!"