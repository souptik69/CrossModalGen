#!/bin/bash

#SBATCH --job-name=AVVP_train
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu006
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=165:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/011_AVVP_Novel_VA_Test/avvp_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/011_AVVP_Novel_VA_Test/avvp_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/011_AVVP_Novel_VA_Test

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run AVVP training
python $SRCDIR/CMG_trial1/src/avvp_novel.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Novel_Model_Final/AVT_Tests/AVVP/noCascade/90k/OldDecoder/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --model_save_path "$SRCDIR/CMG_trial1/Novel_Model_Final/AVT_Tests/AVVP/noCascade/90k/OldDecoder/V2A/" \
    --print_freq 1 \
    --eval_freq 1

echo "AVVP training complete!"